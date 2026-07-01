"""UI: группировка и семантическая кластеризация ошибок бенчмарка."""

from __future__ import annotations

import json
import os
from typing import Sequence

import pandas as pd
import streamlit as st

from dialog_clustering import run_dialog_bertopic_clustering
from ui.dialog_results_ui import format_goals_for_result_row
from analysis.error_clustering import (
    CUSTOM_RECORD_FIELD_SENTINEL,
    SUGGESTED_ERROR_CLUSTER_FIELDS,
    failure_groups_summary_rows,
    failures_fingerprint,
    field_label_for_ui,
    group_failures_by_field,
    is_history_cluster_field,
    resolve_error_cluster_field,
)
from ui.clustering_ui import (
    CLUSTERING_ALGORITHM_LABEL,
    bertopic_result_from_run,
    render_clustering_pipeline_ui,
    render_clustering_results_panel,
)
from common.security import redact_secrets_for_display


def clear_error_cluster_session(key_prefix: str) -> None:
    """Сбрасывает кэш виджетов и результатов кластеризации для блока анализа ошибок."""
    for key in list(st.session_state.keys()):
        if key.startswith(key_prefix):
            st.session_state.pop(key, None)


def render_failure_clustering_section(
    failures: Sequence[dict],
    *,
    key_prefix: str = "results_err_cluster",
    data_fingerprint: str = "",
) -> None:
    if not failures:
        return

    fp_store_key = f"{key_prefix}_results_fp"
    if data_fingerprint and st.session_state.get(fp_store_key) != data_fingerprint:
        clear_error_cluster_session(key_prefix)
        st.session_state[fp_store_key] = data_fingerprint

    st.subheader("Анализ ошибок")
    st.caption(
        f"Анализ **{len(failures)}** кейсов с `accuracy = 0`. "
        "Выберите поле: для `reason` и других скалярных полей каждое **уникальное значение** — отдельная группа."
    )

    method = st.radio(
        "Способ",
        options=["По значению поля", "Кластеризация"],
        horizontal=True,
        key=f"{key_prefix}_method",
        help=(
            "**По значению поля** — точное совпадение текста (например, каждый distinct `reason`). "
            "**Кластеризация** — объединяет похожие формулировки в темы."
        ),
    )

    field_keys = [k for k, _ in SUGGESTED_ERROR_CLUSTER_FIELDS]
    field_labels = {k: lbl for k, lbl in SUGGESTED_ERROR_CLUSTER_FIELDS}

    col_f, col_t = st.columns([2, 1])
    with col_f:
        field_sel = st.selectbox(
            "Поле для группировки",
            options=field_keys,
            format_func=lambda k: field_labels.get(k, k),
            key=f"{key_prefix}_group_field",
        )
    custom_field = ""
    if field_sel == CUSTOM_RECORD_FIELD_SENTINEL:
        custom_field = st.text_input(
            "Имя поля (верхний уровень JSONL или `context.category`)",
            placeholder="reason",
            key=f"{key_prefix}_group_custom_field",
        )

    turn_index = 0
    resolved_preview = field_sel if field_sel != CUSTOM_RECORD_FIELD_SENTINEL else custom_field
    root_field = (resolved_preview or "").split(".", 1)[0]
    if is_history_cluster_field(root_field):
        turn_index = st.number_input(
            "Номер реплики в history (0 — весь текст)",
            min_value=0,
            max_value=50,
            value=0,
            key=f"{key_prefix}_group_turn",
        )

    try:
        cluster_field = resolve_error_cluster_field(field_sel, custom_field)
    except ValueError as exc:
        st.warning(str(exc))
        return

    fp = failures_fingerprint(failures, cluster_field, int(turn_index))

    if method == "По значению поля":
        _render_exact_groups(
            failures,
            cluster_field=cluster_field,
            turn_index=int(turn_index),
            key_prefix=key_prefix,
            fingerprint=fp,
        )
    else:
        _render_bertopic_clusters(
            failures,
            cluster_field=cluster_field,
            turn_index=int(turn_index),
            key_prefix=key_prefix,
            fingerprint=fp,
        )


def _render_exact_groups(
    failures: Sequence[dict],
    *,
    cluster_field: str,
    turn_index: int,
    key_prefix: str,
    fingerprint: str,
) -> None:
    groups = group_failures_by_field(
        failures,
        cluster_field,
        turn_index=turn_index,
    )
    if not groups:
        st.info("Нет групп для отображения.")
        return

    n_unique = len(groups)
    st.metric("Групп (уникальных значений)", n_unique)

    summary = failure_groups_summary_rows(
        groups,
        field=cluster_field,
        total_failures=len(failures),
    )
    st.dataframe(
        pd.DataFrame(summary),
        use_container_width=True,
        hide_index=True,
    )

    max_groups = st.slider(
        "Показать детали для первых N групп",
        min_value=1,
        max_value=min(30, n_unique),
        value=min(10, n_unique),
        key=f"{key_prefix}_exact_top_n",
    )

    for i, grp in enumerate(groups[:max_groups], start=1):
        title_val = grp.value
        if len(title_val) > 80:
            title_val = title_val[:77] + "…"
        with st.expander(
            f"#{i} · {grp.count} кейс(ов) · {field_label_for_ui(cluster_field)}: {title_val}",
            expanded=False,
        ):
            st.text(grp.value if len(grp.value) <= 4000 else grp.value[:4000] + "…")
            rows = []
            for rec in grp.records:
                rows.append(
                    {
                        "dialog_id": rec.get("dialog_id", "—"),
                        "turns": rec.get("turns", "—"),
                        "goals": format_goals_for_result_row(rec)[:200],
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            ndjson = "\n".join(
                json.dumps(redact_secrets_for_display(r), ensure_ascii=False)
                for r in grp.records
            )
            safe_name = "".join(
                c if c.isalnum() or c in "._-" else "_"
                for c in title_val[:30]
            )
            st.download_button(
                "Скачать группу (JSONL)",
                ndjson,
                file_name=f"failures_{cluster_field.replace('.', '_')}_{safe_name}.jsonl",
                mime="application/json",
                key=f"{key_prefix}_dl_{fingerprint}_{i}",
            )

    all_ndjson = "\n".join(
        json.dumps(
            {
                "cluster_field": cluster_field,
                "cluster_value": grp.value,
                **redact_secrets_for_display(dict(rec)),
            },
            ensure_ascii=False,
        )
        for grp in groups
        for rec in grp.records
    )
    st.download_button(
        "Скачать все группы с меткой поля (JSONL)",
        all_ndjson,
        file_name=f"failures_by_{cluster_field.replace('.', '_')}.jsonl",
        mime="application/json",
        key=f"{key_prefix}_dl_all_{fingerprint}",
    )


def _render_bertopic_clusters(
    failures: Sequence[dict],
    *,
    cluster_field: str,
    turn_index: int,
    key_prefix: str,
    fingerprint: str,
) -> None:
    result_key = f"{key_prefix}_bertopic_result"
    fp_key = f"{key_prefix}_bertopic_fp"

    stored_fp = st.session_state.get(fp_key)
    if stored_fp and stored_fp != fingerprint:
        st.warning("Набор ошибок или поле изменились — перезапустите кластеризацию.")

    cfg = render_clustering_pipeline_ui(
        f"{key_prefix}_bertopic",
        bertopic=True,
        supported_models=None,
        fixed_cluster_field=cluster_field,
        fixed_cluster_turn_index=int(turn_index),
    )

    if st.button("Запустить кластеризацию", type="primary", key=f"{key_prefix}_bertopic_run"):
        progress = st.progress(0.0, text="Старт…")
        status = st.empty()

        def on_progress(msg: str, frac: float) -> None:
            progress.progress(min(1.0, max(0.0, frac)), text=msg)
            status.caption(msg)

        try:
            result = run_dialog_bertopic_clustering(
                list(failures),
                cluster_field=cluster_field,
                output_field="cluster_label",
                umap_settings=cfg["umap_settings"],
                bertopic_settings=cfg["bertopic_settings"],
                with_llm=cfg["with_llm"],
                llm_model=cfg["llm_model"] if cfg["with_llm"] else "",
                llm_api_key=os.getenv("LITELLM_API_KEY", ""),
                llm_api_base=os.getenv("LITELLM_API_BASE", ""),
                llm_sample_dialogs=int(cfg["llm_sample"]),
                llm_prompt_template=cfg["llm_prompt_template"],
                dedup_enabled=cfg["dedup_enabled"],
                dedup_field=cfg["dedup_field"],
                dedup_turn_index=cfg["dedup_turn_index"],
                cluster_turn_index=int(turn_index),
                dedup_similarity_threshold=float(cfg["dedup_threshold"]),
                tfidf_top_n=cfg["tfidf_top_n"],
                on_progress=on_progress,
            )
            st.session_state[result_key] = bertopic_result_from_run(result, cfg)
            st.session_state[fp_key] = fingerprint
            progress.progress(1.0, text="Готово")
            status.success(f"Тем: **{result.n_clusters}**")
            st.rerun()
        except Exception as exc:
            progress.empty()
            status.empty()
            st.error(f"Ошибка кластеризации: {exc}")

    stored = st.session_state.get(result_key)
    if not stored or st.session_state.get(fp_key) != fingerprint:
        return

    records = stored.get("records") or []
    if not records:
        return

    render_clustering_results_panel(
        stored,
        key_prefix=f"{key_prefix}_panel",
        algorithm_name=CLUSTERING_ALGORITHM_LABEL,
        download_prefix="failures_clustered",
        show_balanced_sample=False,
        similar_query_label="Поле для сравнения",
    )
