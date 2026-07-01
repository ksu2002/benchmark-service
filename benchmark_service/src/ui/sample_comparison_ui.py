"""UI Streamlit для сравнения выборок."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
import streamlit as st

from dialog_clustering import load_jsonl_records
from analysis.sample_comparison import compare_samples, goal_from_record, cluster_id_from_record
from ui.sample_storage_ui import decode_uploaded_text, render_load_saved_samples_ui


@dataclass
class ComparisonSourceOption:
    """Источник записей для одной стороны сравнения."""

    option_id: str
    label: str
    records: List[dict]
    record_embeddings: Optional[Sequence[Optional[Sequence[float]]]] = None
    embedding_source_records: Optional[Sequence[Mapping[str, Any]]] = None
    output_field: str = "cluster_label"


def load_comparison_records_from_text(text: str) -> tuple[List[dict], int]:
    records, bad = load_jsonl_records(text)
    return [dict(r) for r in records], bad


def _store_loaded_comparison_side(
    *,
    records_key: str,
    label_key: str,
    records: List[dict],
    label: str,
) -> None:
    st.session_state[records_key] = records
    st.session_state[label_key] = label


def _comparison_opts_for_sources(
    source_a: ComparisonSourceOption,
    source_b: ComparisonSourceOption,
) -> dict:
    opts: dict = {"output_field": source_a.output_field or source_b.output_field}
    if source_a.record_embeddings is not None:
        opts["record_embeddings"] = source_a.record_embeddings
        opts["embedding_source_records"] = (
            source_a.embedding_source_records or source_a.records
        )
    elif source_b.record_embeddings is not None:
        opts["record_embeddings"] = source_b.record_embeddings
        opts["embedding_source_records"] = (
            source_b.embedding_source_records or source_b.records
        )
    return opts


def render_comparison_side_selector(
    *,
    key_prefix: str,
    title: str,
    preset_options: Sequence[ComparisonSourceOption],
    loaded_records_key: str,
    loaded_label_key: str,
    select_key: str,
) -> ComparisonSourceOption:
    """Выбор источника для одной стороны: пресет страницы или загруженный JSONL."""
    st.markdown(f"**{title}**")

    options: Dict[str, ComparisonSourceOption] = {
        opt.option_id: opt for opt in preset_options if opt.records
    }
    loaded = st.session_state.get(loaded_records_key)
    loaded_label = str(st.session_state.get(loaded_label_key) or "Загруженная выборка")
    if loaded:
        options["custom_loaded"] = ComparisonSourceOption(
            option_id="custom_loaded",
            label=loaded_label,
            records=list(loaded),
        )

    option_ids = list(options.keys()) + ["__upload__"]
    if st.session_state.get(select_key) not in option_ids:
        st.session_state[select_key] = option_ids[0] if options else "__upload__"

    picked_id = st.selectbox(
        "Источник",
        options=option_ids,
        format_func=lambda oid: (
            "Загрузить файл / из БД…"
            if oid == "__upload__"
            else options[oid].label
        ),
        key=select_key,
    )

    if picked_id == "__upload__":
        with st.expander("Загрузка выборки", expanded=True):

            def _on_text(text: str, *, name: str) -> None:
                records, bad = load_comparison_records_from_text(text)
                if not records:
                    st.error("В файле не найдено записей.")
                    return
                _store_loaded_comparison_side(
                    records_key=loaded_records_key,
                    label_key=loaded_label_key,
                    records=records,
                    label=name,
                )
                st.session_state[select_key] = "custom_loaded"
                if bad:
                    st.warning(f"Пропущено некорректных строк: **{bad}**.")
                st.rerun()

            def _on_file_text(text: str) -> None:
                fname = st.session_state.get(f"{key_prefix}_upload_name") or "Файл"
                _on_text(text, name=fname)

            def _on_db(raw: bytes, row: dict) -> None:
                name = str(row.get("name") or "Выборка из БД")
                _on_text(raw.decode("utf-8", errors="replace"), name=name)

            src = st.radio(
                "Откуда загрузить",
                ["Файл JSONL", "Сохранённые выборки"],
                horizontal=True,
                key=f"{key_prefix}_upload_src",
            )
            if src == "Файл JSONL":
                uploaded = st.file_uploader(
                    "Файл .jsonl",
                    type=["jsonl", "json", "txt"],
                    key=f"{key_prefix}_upload",
                )
                if uploaded is not None:
                    st.session_state[f"{key_prefix}_upload_name"] = uploaded.name
                    st.caption(f"**{uploaded.name}** ({uploaded.size} байт)")
                    if st.button("Загрузить файл", key=f"{key_prefix}_upload_btn"):
                        try:
                            _on_file_text(
                                decode_uploaded_text(uploaded, support_json_array=True)
                            )
                        except Exception as e:
                            st.error(str(e))
            else:
                render_load_saved_samples_ui(
                    key_prefix=key_prefix,
                    on_load=_on_db,
                    show_delete=False,
                    caption="Выборки из Postgres + MinIO.",
                    select_key=f"{key_prefix}_saved_select",
                    load_key=f"{key_prefix}_saved_load",
                )

        if loaded:
            return options["custom_loaded"]
        return ComparisonSourceOption(
            option_id="__empty__",
            label="—",
            records=[],
        )

    return options[picked_id]


def render_dual_source_comparison(
    preset_options: Sequence[ComparisonSourceOption],
    *,
    key_prefix: str,
    default_a_id: Optional[str] = None,
    default_b_id: Optional[str] = None,
) -> None:
    """
    Сравнение двух выборок с выбором источника для каждой стороны
    (пресеты страницы или загрузка JSONL / из БД).
    """
    valid_presets = [o for o in preset_options if o.records]
    if not valid_presets:
        st.caption(
            "Пресеты с этой страницы недоступны — выберите **«Загрузить файл / из БД…»** "
            "для каждой стороны."
        )

    select_a_key = f"{key_prefix}_side_a"
    select_b_key = f"{key_prefix}_side_b"
    preset_ids = {o.option_id for o in valid_presets}
    if default_a_id and select_a_key not in st.session_state and default_a_id in preset_ids:
        st.session_state[select_a_key] = default_a_id
    if default_b_id and select_b_key not in st.session_state and default_b_id in preset_ids:
        st.session_state[select_b_key] = default_b_id

    col_a, col_b = st.columns(2)
    with col_a:
        source_a = render_comparison_side_selector(
            key_prefix=f"{key_prefix}_a",
            title="Выборка A (базовая)",
            preset_options=valid_presets,
            loaded_records_key=f"{key_prefix}_loaded_a",
            loaded_label_key=f"{key_prefix}_loaded_a_label",
            select_key=select_a_key,
        )
    with col_b:
        source_b = render_comparison_side_selector(
            key_prefix=f"{key_prefix}_b",
            title="Выборка B (текущая)",
            preset_options=valid_presets,
            loaded_records_key=f"{key_prefix}_loaded_b",
            loaded_label_key=f"{key_prefix}_loaded_b_label",
            select_key=select_b_key,
        )

    if not source_a.records:
        st.info("Выберите или загрузите **выборку A**.")
        return
    if not source_b.records:
        st.info("Выберите или загрузите **выборку B**.")
        return

    opts = _comparison_opts_for_sources(source_a, source_b)
    render_sample_comparison_panel(
        source_a.records,
        source_b.records,
        key_prefix=f"{key_prefix}_cmp",
        baseline_label=source_a.label,
        current_label=source_b.label,
        in_expander=False,
        title="",
        **opts,
    )


def _distribution_table(
    records: Sequence[Mapping[str, Any]],
    *,
    axis: str,
    output_field: str = "cluster_label",
) -> pd.DataFrame:
    if axis == "goal":
        counts = Counter(goal_from_record(r) for r in records)
        rows = [{"Категория": k, "Диалогов": v} for k, v in counts.most_common()]
    elif axis == "cluster":
        counts: Counter = Counter()
        labels: dict = {}
        for r in records:
            cid = cluster_id_from_record(r)
            if cid is None or cid < 0:
                continue
            counts[cid] += 1
            if cid not in labels:
                lbl = str(r.get(output_field) or "").strip() or f"Кластер {cid}"
                labels[cid] = lbl
        rows = [
            {
                "cluster_id": cid,
                "Кластер": labels.get(cid, str(cid)),
                "Диалогов": counts[cid],
            }
            for cid in sorted(counts.keys())
        ]
    else:
        rows = []
    return pd.DataFrame(rows)


def _render_sample_comparison_body(
    baseline_records: Sequence[Mapping[str, Any]],
    current_records: Sequence[Mapping[str, Any]],
    *,
    baseline_label: str,
    current_label: str,
    record_embeddings: Optional[Sequence[Optional[Sequence[float]]]] = None,
    embedding_source_records: Optional[Sequence[Mapping[str, Any]]] = None,
    output_field: str = "cluster_label",
) -> None:
    cmp = compare_samples(
        baseline_records,
        current_records,
        baseline_label=baseline_label,
        current_label=current_label,
        record_embeddings=record_embeddings,
        embedding_source_records=embedding_source_records,
    )

    st.caption(
        f"**{baseline_label}** — {cmp.baseline.n_dialogs} диалогов · "
        f"**{current_label}** — {cmp.current.n_dialogs} диалогов. "
        "↑ в колонке Δ — изменение в «лучшую» сторону для метрики."
    )

    if (
        cmp.baseline.n_dialogs == cmp.current.n_dialogs
        and cmp.baseline.to_dict() == cmp.current.to_dict()
    ):
        st.info("Выборки совпадают по всем метрикам.")

    df = pd.DataFrame(cmp.comparison_rows())
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    has_clusters = cmp.baseline.n_clusters > 0 or cmp.current.n_clusters > 0
    tab_goal, tab_cluster = st.tabs(["По целям", "По кластерам"])

    with tab_goal:
        g1, g2 = st.columns(2)
        with g1:
            st.markdown(f"**{baseline_label}**")
            st.dataframe(
                _distribution_table(baseline_records, axis="goal"),
                use_container_width=True,
                hide_index=True,
            )
        with g2:
            st.markdown(f"**{current_label}**")
            st.dataframe(
                _distribution_table(current_records, axis="goal"),
                use_container_width=True,
                hide_index=True,
            )

    with tab_cluster:
        if not has_clusters:
            st.caption("Кластеры доступны после кластеризации.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{baseline_label}**")
                st.dataframe(
                    _distribution_table(
                        baseline_records,
                        axis="cluster",
                        output_field=output_field,
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            with c2:
                st.markdown(f"**{current_label}**")
                st.dataframe(
                    _distribution_table(
                        current_records,
                        axis="cluster",
                        output_field=output_field,
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


def render_sample_comparison_panel(
    baseline_records: Sequence[Mapping[str, Any]],
    current_records: Sequence[Mapping[str, Any]],
    *,
    key_prefix: str,
    baseline_label: str = "Исходная",
    current_label: str = "Текущая",
    record_embeddings: Optional[Sequence[Optional[Sequence[float]]]] = None,
    embedding_source_records: Optional[Sequence[Mapping[str, Any]]] = None,
    output_field: str = "cluster_label",
    expanded: bool = True,
    in_expander: bool = True,
    title: str = "📊 Сравнение: исходная vs текущая выборка",
) -> None:
    del key_prefix  # зарезервировано для будущих виджетов
    if not baseline_records:
        return
    if not current_records:
        st.info("Текущая выборка пуста — нечего сравнивать.")
        return

    kwargs = dict(
        baseline_label=baseline_label,
        current_label=current_label,
        record_embeddings=record_embeddings,
        embedding_source_records=embedding_source_records,
        output_field=output_field,
    )

    if in_expander:
        with st.expander(title, expanded=expanded):
            _render_sample_comparison_body(
                baseline_records, current_records, **kwargs
            )
    else:
        if title:
            st.subheader(title)
        _render_sample_comparison_body(
            baseline_records, current_records, **kwargs
        )
