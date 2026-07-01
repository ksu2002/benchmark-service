"""UI-компоненты кластеризации диалогов для Streamlit."""

from __future__ import annotations

import json
import os
from typing import Callable, Optional, Sequence

import streamlit as st

from dialog_clustering import (
    BertopicSettings,
    CUSTOM_RECORD_FIELD_SENTINEL,
    DEFAULT_CLUSTER_LLM_PROMPT_TEMPLATE,
    HdbscanSettings,
    RECORD_FIELD_SELECT_LABELS,
    UmapSettings,
    CLUSTERING_ALGORITHM_LABEL,
    _dialog_id_from_record,
    build_balanced_cluster_sample,
    cluster_example_rows,
    cluster_size_chart_rows,
    cluster_summary_rows,
    export_jsonl_with_stats,
    field_scope_label,
    format_dialog_history_only,
    highlight_keywords_in_html,
    is_history_cluster_field,
    parse_cluster_tfidf_words,
    embed_similarity_query_text,
    extract_cluster_text,
    prepare_similarity_query_text,
    record_similarity_label,
    resolve_record_field,
    run_dialog_bertopic_clustering,
    serialize_cluster_tfidf_words,
    similar_dialog_search_rows_by_embedding,
    turn_scope_label,
)
from ui.dialog_results_ui import render_dialog_length_distribution_panel
from ui.sample_storage_ui import render_export_jsonl_actions

STANDARD_RECORD_FIELD_OPTIONS = ("dialog", "assistant", "user")


def _display_algorithm_name(stored: dict, *, fallback: str = "HDBSCAN") -> str:
    name = str(stored.get("algorithm") or fallback)
    if name == "BERTopic":
        return CLUSTERING_ALGORITHM_LABEL
    return name


def _default_supported_models() -> list[str]:
    try:
        from integrations.litellm import get_model_names

        return get_model_names()
    except Exception:
        return [os.getenv("LITELLM_MODEL_NAME", "openai/gpt-4o-mini")]


def render_cluster_visualizations(stored: dict, *, key_prefix: str = "judge_clust") -> None:
    viz_points = stored.get("viz_points") or []
    records = stored.get("records") or []
    if not viz_points:
        st.info("Нет координат для визуализации (возможно, не получены эмбеддинги).")
        return

    cluster_field = stored.get("cluster_field", "dialog")
    output_field = stored.get("output_field", "cluster_label")
    algorithm = _display_algorithm_name(stored)

    st.markdown("#### Визуализация")
    st.caption(
        f"2D-проекция **UMAP** для обзора. Цвет — `cluster_id`. "
        f"Это отдельная проекция, не обязательно совпадает с пространством {algorithm}."
    )

    hide_noise = st.checkbox(
        "Скрыть шум (cluster_id = -1)",
        value=False,
        key=f"{key_prefix}_viz_hide_noise",
    )
    points = [
        p for p in viz_points if not (hide_noise and int(p.get("cluster_id", 0)) == -1)
    ]
    if not points:
        st.warning("После фильтрации не осталось точек для графика.")
        return

    try:
        import pandas as pd
        import plotly.express as px

        df_viz = pd.DataFrame(points)
        fig_scatter = px.scatter(
            df_viz,
            x="viz_x",
            y="viz_y",
            color="cluster_display",
            hover_data={
                "dialog_id": True,
                "cluster_label": True,
                "snippet": True,
                "cluster_id": True,
                "viz_x": ":.3f",
                "viz_y": ":.3f",
                "cluster_display": False,
            },
            labels={
                "viz_x": "UMAP 1",
                "viz_y": "UMAP 2",
                "cluster_display": "Кластер",
                "snippet": "диалог",
            },
            title="Диалоги на плоскости UMAP",
        )
        fig_scatter.update_layout(
            legend={"itemsizing": "constant"},
            height=520,
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
        )
        fig_scatter.update_traces(marker={"size": 8, "opacity": 0.75})
        st.plotly_chart(fig_scatter, use_container_width=True)

        size_rows = cluster_size_chart_rows(records, output_field=output_field)
        if size_rows:
            st.markdown("#### Размеры кластеров")
            df_sizes = pd.DataFrame(size_rows)
            fig_bar = px.bar(
                df_sizes,
                x="кластер",
                y="диалогов",
                hover_data=["cluster_id", "доля, %"],
                title="Число диалогов в кластере",
                color="cluster_id",
                color_continuous_scale="Blues",
            )
            fig_bar.update_layout(height=420, xaxis={"tickangle": -35})
            st.plotly_chart(fig_bar, use_container_width=True)
    except ImportError:
        import pandas as pd

        st.warning("Установите plotly для интерактивных графиков: `pip install plotly`")
        df_viz = pd.DataFrame(points)
        st.scatter_chart(df_viz, x="viz_x", y="viz_y", color="cluster_display")
        size_rows = cluster_size_chart_rows(records, output_field=output_field)
        if size_rows:
            st.markdown("#### Размеры кластеров")
            st.bar_chart(pd.DataFrame(size_rows).set_index("кластер")["диалогов"])

    st.markdown("#### Примеры диалогов в кластере")
    cluster_options = sorted(
        {int(p.get("cluster_id", -99)) for p in viz_points},
        key=lambda c: (c < 0, -sum(1 for x in viz_points if int(x.get("cluster_id", -99)) == c), c),
    )
    picked_cid = st.selectbox(
        "Кластер",
        options=cluster_options,
        format_func=lambda cid: next(
            (
                p.get("cluster_display") or str(cid)
                for p in viz_points
                if int(p.get("cluster_id", -99)) == cid
            ),
            str(cid),
        ),
        key=f"{key_prefix}_viz_pick_cluster",
    )
    examples_page_size = 5
    total_in_cluster = sum(
        1 for r in records if int(r.get("cluster_id", -99)) == picked_cid
    )
    total_pages = max(1, (total_in_cluster + examples_page_size - 1) // examples_page_size)
    page_key = f"{key_prefix}_viz_examples_page"
    last_cluster_key = f"{key_prefix}_viz_examples_last_cluster"
    if st.session_state.get(last_cluster_key) != picked_cid:
        st.session_state[page_key] = 0
        st.session_state[last_cluster_key] = picked_cid
    page = max(0, min(int(st.session_state.get(page_key, 0)), total_pages - 1))
    st.session_state[page_key] = page

    examples = cluster_example_rows(
        records,
        picked_cid,
        cluster_field=cluster_field,
        output_field=output_field,
        limit=examples_page_size,
        offset=page * examples_page_size,
    )
    cluster_tfidf_words = parse_cluster_tfidf_words(stored.get("cluster_tfidf_words"))
    tfidf_words = cluster_tfidf_words.get(int(picked_cid), [])
    keyword_list = [word for word, _score in tfidf_words]

    if tfidf_words:
        st.markdown(
            "**Ключевые слова TF-IDF:** "
            + ", ".join(f"**{word}** ({score:.3f})" for word, score in tfidf_words)
        )
    elif int(picked_cid) >= 0:
        st.caption("Ключевые слова TF-IDF для этого кластера не найдены.")

    if examples:
        table_rows = []
        for row in examples:
            dialog_html = highlight_keywords_in_html(
                str(row.get("history") or ""),
                keyword_list,
            )
            table_rows.append(
                "<tr>"
                f"<td style='vertical-align:top;white-space:nowrap;padding:6px 10px'>"
                f"{row['dialog_id']}</td>"
                f"<td style='vertical-align:top;padding:6px 10px;line-height:1.45'>"
                f"{dialog_html}</td>"
                "</tr>"
            )
        table_html = (
            "<table style='width:100%;border-collapse:collapse;font-size:0.92rem'>"
            "<thead><tr>"
            "<th style='text-align:left;padding:6px 10px;border-bottom:1px solid #ddd'>ID</th>"
            "<th style='text-align:left;padding:6px 10px;border-bottom:1px solid #ddd'>Диалог</th>"
            "</tr></thead><tbody>"
            + "".join(table_rows)
            + "</tbody></table>"
        )
        st.markdown(table_html, unsafe_allow_html=True)
        if total_pages > 1:
            nav_cols = st.columns([1, 3, 1])
            with nav_cols[0]:
                if st.button(
                    "← Назад",
                    key=f"{key_prefix}_viz_examples_prev",
                    disabled=page <= 0,
                ):
                    st.session_state[page_key] = page - 1
                    st.rerun()
            with nav_cols[1]:
                shown_from = page * examples_page_size + 1
                shown_to = min((page + 1) * examples_page_size, total_in_cluster)
                st.caption(
                    f"Страница **{page + 1}** из **{total_pages}** · "
                    f"показано **{shown_from}–{shown_to}** из **{total_in_cluster}** диалогов"
                )
            with nav_cols[2]:
                if st.button(
                    "Вперёд →",
                    key=f"{key_prefix}_viz_examples_next",
                    disabled=page >= total_pages - 1,
                ):
                    st.session_state[page_key] = page + 1
                    st.rerun()
        elif total_in_cluster > 0:
            st.caption(f"Показано **{len(examples)}** из **{total_in_cluster}** диалогов.")
    else:
        st.info("В выбранном кластере нет диалогов.")


def render_similar_dialogs_block(
    stored: dict,
    *,
    key_prefix: str,
    query_label: str = "Эталонный диалог",
) -> None:
    """Поиск похожих диалогов по эмбеддингам из шага кластеризации."""
    records = stored.get("records") or []
    embeddings = stored.get("record_embeddings") or []
    if not records or not embeddings or len(records) != len(embeddings):
        return

    valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
    if len(valid_indices) < 2:
        return

    output_field = stored.get("output_field", "cluster_label")
    cluster_field = stored.get("cluster_field", "dialog")
    turn_index = int(stored.get("cluster_turn_index") or 0)

    st.markdown("---")
    st.subheader("Похожие диалоги")
    st.caption(
        "Cosine similarity по эмбеддингам из кластеризации "
        f"({field_scope_label(cluster_field, turn_index)}). "
        "Введите свой текст или подставьте диалог из выборки — "
        "для эталона нужен один запрос к API эмбеддингов."
    )

    fill_key = f"{key_prefix}_similar_fill_idx"
    text_key = f"{key_prefix}_similar_query_text"
    prev_fill_key = f"{key_prefix}_similar_prev_fill"
    emb_text_key = f"{key_prefix}_similar_emb_for"
    emb_vec_key = f"{key_prefix}_similar_emb_vec"

    if fill_key not in st.session_state or st.session_state[fill_key] not in valid_indices:
        st.session_state[fill_key] = valid_indices[0]
    if text_key not in st.session_state:
        st.session_state[text_key] = extract_cluster_text(
            records[valid_indices[0]], cluster_field, turn_index=turn_index
        )
    if prev_fill_key not in st.session_state:
        st.session_state[prev_fill_key] = st.session_state[fill_key]

    fill_col, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with fill_col:
        picked = st.selectbox(
            "Подставить из диалога",
            options=valid_indices,
            format_func=lambda i: record_similarity_label(
                records[i], output_field=output_field
            ),
            key=fill_key,
        )
        if st.session_state[prev_fill_key] != picked:
            st.session_state[text_key] = extract_cluster_text(
                records[picked], cluster_field, turn_index=turn_index
            )
            st.session_state[prev_fill_key] = picked

    with ctrl2:
        top_k = st.number_input(
            "Топ-K",
            min_value=1,
            max_value=min(50, len(valid_indices)),
            value=min(10, len(valid_indices)),
            key=f"{key_prefix}_similar_top_k",
        )
    with ctrl3:
        min_sim = st.slider(
            "Мин. similarity",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.01,
            key=f"{key_prefix}_similar_min_sim",
        )

    query_text = st.text_area(
        query_label,
        key=text_key,
        height=120,
        help=(
            "Текст для поиска похожих. Можно редактировать подставленное значение "
            "или ввести произвольный текст."
        ),
    )

    include_self = st.checkbox(
        "Показать эталон в списке",
        value=False,
        key=f"{key_prefix}_similar_include_self",
    )

    model_text = prepare_similarity_query_text(query_text, cluster_field=cluster_field)
    if not model_text:
        st.warning(f"Заполните поле «{query_label}».")
        return

    if (
        st.session_state.get(emb_text_key) != model_text
        or not st.session_state.get(emb_vec_key)
    ):
        with st.spinner("Эмбеддинг эталонного текста…"):
            query_emb = embed_similarity_query_text(
                query_text, cluster_field=cluster_field
            )
        st.session_state[emb_text_key] = model_text
        st.session_state[emb_vec_key] = query_emb
    else:
        query_emb = st.session_state[emb_vec_key]

    if query_emb is None:
        st.error("Не удалось получить эмбеддинг для эталонного текста.")
        return

    exclude_index: Optional[int] = None
    if not include_self:
        filled_text = extract_cluster_text(
            records[picked], cluster_field, turn_index=turn_index
        )
        if filled_text.strip() == (query_text or "").strip():
            exclude_index = int(picked)

    hits = similar_dialog_search_rows_by_embedding(
        records,
        embeddings,
        query_emb,
        output_field=output_field,
        top_k=int(top_k),
        min_similarity=float(min_sim),
        exclude_index=exclude_index,
    )

    if not hits:
        st.info(
            f"Нет диалогов с similarity ≥ **{min_sim:.2f}**. "
            "Понизьте порог или измените эталонный текст."
        )
        return

    import pandas as pd

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "similarity": h["similarity"],
                    "dialog_id": h["dialog_id"],
                    "кластер": h["кластер"],
                    "превью": h["превью"],
                }
                for h in hits
            ]
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "similarity": st.column_config.NumberColumn(
                "similarity",
                format="%.3f",
                help="Cosine similarity к эталонному диалогу.",
            ),
            "превью": st.column_config.TextColumn("превью", width="large"),
        },
    )

    with st.expander("Полные тексты похожих диалогов", expanded=False):
        for h in hits:
            st.markdown(
                f"**{h['similarity']:.3f}** · `{h['dialog_id']}` · {h['кластер']}"
            )
            st.text(format_dialog_history_only(records[h["_index"]]))
            st.markdown("---")


def render_balanced_sample_coverage_viz(
    stored: dict,
    bal_records: list,
    *,
    key_prefix: str,
) -> None:
    """UMAP: все диалоги (фон) и точки сбалансированной выборки (передний план)."""
    viz_points = stored.get("viz_points") or []
    if not viz_points or not bal_records:
        return

    selected_ids = {_dialog_id_from_record(r) for r in bal_records}
    background: list[dict] = []
    selected: list[dict] = []
    for point in viz_points:
        if str(point.get("dialog_id") or "") in selected_ids:
            selected.append(point)
        else:
            background.append(point)

    if not selected:
        st.info("Нет координат UMAP для точек сбалансированной выборки.")
        return

    all_cluster_ids = {
        int(p.get("cluster_id", -2))
        for p in viz_points
        if int(p.get("cluster_id", -2)) >= 0
    }
    sample_cluster_ids = {
        int(r.get("cluster_id", -2))
        for r in bal_records
        if int(r.get("cluster_id", -2)) >= 0
    }
    hide_noise_bg = st.checkbox(
        "Скрыть шум на фоне (cluster_id = -1)",
        value=False,
        key=f"{key_prefix}_bal_viz_hide_noise",
    )
    if hide_noise_bg:
        background = [
            p for p in background if int(p.get("cluster_id", -2)) != -1
        ]

    st.markdown("#### Покрытие пространства")
    st.caption(
        "Серые точки — все диалоги после кластеризации, цветные — сбалансированная выборка. "
        "Хорошее покрытие: выборка затрагивает разные области облака и разные кластеры."
    )
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("В выборке", len(selected))
    with m2:
        st.metric("На фоне", len(background))
    with m3:
        if all_cluster_ids:
            st.metric(
                "Кластеров ≥ 0",
                f"{len(sample_cluster_ids)}/{len(all_cluster_ids)}",
            )
        else:
            st.metric("Кластеров ≥ 0", len(sample_cluster_ids))

    try:
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go

        fig = go.Figure()
        if background:
            fig.add_trace(
                go.Scatter(
                    x=[p["viz_x"] for p in background],
                    y=[p["viz_y"] for p in background],
                    mode="markers",
                    name=f"Все диалоги ({len(background)})",
                    marker={"size": 5, "color": "#d0d0d0", "opacity": 0.35},
                    hoverinfo="skip",
                )
            )

        df_sel = pd.DataFrame(selected)
        fig_sel = px.scatter(
            df_sel,
            x="viz_x",
            y="viz_y",
            color="cluster_display",
            hover_data={
                "dialog_id": True,
                "cluster_label": True,
                "snippet": True,
                "cluster_id": True,
                "viz_x": ":.3f",
                "viz_y": ":.3f",
                "cluster_display": False,
            },
            labels={
                "viz_x": "UMAP 1",
                "viz_y": "UMAP 2",
                "cluster_display": "Кластер",
                "snippet": "диалог",
            },
        )
        for trace in fig_sel.data:
            trace.marker.size = 11
            trace.marker.opacity = 0.9
            fig.add_trace(trace)

        fig.update_layout(
            title="Сбалансированная выборка на фоне всех диалогов",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            height=520,
            margin={"l": 20, "r": 20, "t": 40, "b": 20},
            legend={"itemsizing": "constant"},
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        import pandas as pd

        st.warning("Установите plotly для интерактивного графика: `pip install plotly`")
        df_all = pd.DataFrame(background + selected)
        st.scatter_chart(
            df_all,
            x="viz_x",
            y="viz_y",
            color="cluster_display" if "cluster_display" in df_all.columns else None,
        )


def render_balanced_sample_block(stored: dict, *, key_prefix: str = "judge_clust") -> None:
    records = stored.get("records") or []
    if not records:
        return

    output_field = stored.get("output_field", "cluster_label")
    algorithm = _display_algorithm_name(stored)
    n_clusters = int(stored.get("n_clusters") or 0)
    if n_clusters == 0 and not any(int(r.get("cluster_id", -2)) == -1 for r in records):
        st.info(
            f"Сначала нужны кластеры с меткой ≥ 0 — уменьшите min_cluster_size / min_topic_size "
            f"или увеличьте датасет."
        )
        return

    st.markdown("---")
    st.subheader("Сбалансированная выборка")
    st.caption(
        "Равномерная подвыборка по кластерам: из каждого **cluster_id ≥ 0** — до N диалогов, "
        "чтобы в итоговом наборе были **разные темы**. "
        "Внутри кластера по умолчанию **MMR** — разнообразная выборка по эмбеддингам "
        "из шага кластеризации (без повторного запроса к API). "
        "Опционально — по одному примеру на каждую уникальную метку среди **шума (-1)**."
    )

    cluster_field = stored.get("cluster_field", "dialog")

    b_cols1 = st.columns(4)
    with b_cols1[0]:
        per_cluster = st.number_input(
            "Из каждого кластера",
            min_value=1,
            max_value=100,
            value=int(st.session_state.get(f"{key_prefix}_bal_per_cluster", 3)),
            key=f"{key_prefix}_bal_per_cluster",
            help="Сколько диалогов взять из каждого cluster_id ≥ 0.",
        )
    with b_cols1[1]:
        total_max = st.number_input(
            "Макс. всего (0 = без лимита)",
            min_value=0,
            max_value=10000,
            value=int(st.session_state.get(f"{key_prefix}_bal_total_max", 0)),
            key=f"{key_prefix}_bal_total_max",
        )
    with b_cols1[2]:
        bal_seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=999999,
            value=int(st.session_state.get(f"{key_prefix}_bal_seed", 42)),
            key=f"{key_prefix}_bal_seed",
        )
    with b_cols1[3]:
        include_noise = st.checkbox(
            "Включить шум (-1)",
            value=bool(st.session_state.get(f"{key_prefix}_bal_include_noise", True)),
            key=f"{key_prefix}_bal_include_noise",
        )

    b_cols2 = st.columns(2)
    with b_cols2[0]:
        per_noise_label = st.number_input(
            "Из каждой метки шума",
            min_value=1,
            max_value=20,
            value=int(st.session_state.get(f"{key_prefix}_bal_per_noise", 1)),
            key=f"{key_prefix}_bal_per_noise",
            disabled=not include_noise,
            help="Для cluster_id = -1: до N диалогов на каждую уникальную cluster_label.",
        )
    with b_cols2[1]:
        st.caption(
            f"Кластеров {algorithm}: **{n_clusters}** · "
            f"шумовых диалогов: **{sum(1 for r in records if int(r.get('cluster_id', -2)) == -1)}**"
        )

    b_cols3 = st.columns(3)
    with b_cols3[0]:
        sample_method = st.selectbox(
            "Выбор внутри кластера",
            options=["mmr", "random"],
            format_func=lambda x: "MMR (разнообразие)" if x == "mmr" else "Случайно",
            index=0,
            key=f"{key_prefix}_bal_method",
            help="MMR: типичные, но не похожие друг на друга диалоги; эмбеддинги берутся из кластеризации.",
        )
    with b_cols3[1]:
        mmr_lambda = st.slider(
            "MMR λ (релевантность)",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get(f"{key_prefix}_bal_mmr_lambda", 0.6)),
            step=0.05,
            key=f"{key_prefix}_bal_mmr_lambda",
            disabled=sample_method != "mmr",
            help="Выше — ближе к центру темы; ниже — сильнее штраф за похожесть на уже выбранные.",
        )
    with b_cols3[2]:
        st.caption(
            f"Текст для MMR: **{field_scope_label(cluster_field, int(stored.get('cluster_turn_index') or 0))}** "
            f"(то же, что при кластеризации)."
        )

    if st.button(
        "Сформировать сбалансированную выборку",
        type="primary",
        key=f"{key_prefix}_bal_build",
    ):
        progress = st.progress(0.0, text="Старт…")
        status = st.empty()

        def on_progress(msg: str, frac: float) -> None:
            progress.progress(min(1.0, max(0.0, frac)), text=msg)
            status.caption(msg)

        try:
            bal = build_balanced_cluster_sample(
                records,
                output_field=output_field,
                cluster_field=cluster_field,
                per_cluster=int(per_cluster),
                per_noise_label=int(per_noise_label),
                include_noise=include_noise,
                total_max=int(total_max) if int(total_max) > 0 else None,
                seed=int(bal_seed),
                sample_method=sample_method,
                mmr_lambda=float(mmr_lambda),
                record_embeddings=stored.get("record_embeddings"),
                on_progress=on_progress if sample_method == "mmr" else None,
            )
            progress.progress(1.0, text="Готово")
            if not bal.records:
                progress.empty()
                status.empty()
                st.warning("Выборка пуста — проверьте параметры и наличие кластеров.")
            else:
                st.session_state[f"{key_prefix}_balanced"] = {
                    "records": bal.records,
                    "stats_rows": bal.stats_rows,
                    "total": bal.total,
                    "groups_used": bal.groups_used,
                    "sample_method": sample_method,
                }
                status.success(
                    f"Сформировано **{bal.total}** диалогов из **{bal.groups_used}** групп "
                    f"(метод: **{sample_method}**)."
                )
                st.rerun()
        except Exception as e:
            progress.empty()
            status.empty()
            st.error(str(e))

    bal_stored = st.session_state.get(f"{key_prefix}_balanced")
    if not bal_stored or not bal_stored.get("records"):
        return

    st.success(
        f"Сформировано **{bal_stored.get('total', 0)}** диалогов "
        f"из **{bal_stored.get('groups_used', 0)}** групп."
    )
    st.dataframe(
        bal_stored.get("stats_rows") or [],
        use_container_width=True,
        hide_index=True,
    )
    render_balanced_sample_coverage_viz(
        stored,
        bal_stored.get("records") or [],
        key_prefix=key_prefix,
    )


def render_clustering_results_panel(
    stored: dict,
    *,
    key_prefix: str,
    algorithm_name: str,
    download_prefix: str,
    show_balanced_sample: bool = True,
    similar_query_label: str = "Эталонный диалог",
) -> None:
    st.markdown("---")
    st.subheader("Результаты")

    n_before = int(stored.get("n_input_before_dedup") or len(stored.get("records") or []))
    dedup_exact = int(stored.get("dedup_exact_removed") or 0)
    dedup_sem = int(stored.get("dedup_semantic_removed") or 0)

    dedup_field_stored = stored.get("dedup_field") or "dialog"
    dedup_turn_stored = int(stored.get("dedup_turn_index") or 0)
    cluster_field_stored = stored.get("cluster_field", "dialog")
    cluster_turn_stored = int(stored.get("cluster_turn_index") or 0)
    st.caption(
        f"Поле для дубликатов: **{field_scope_label(dedup_field_stored, dedup_turn_stored)}** · "
        f"кластеризация: **{field_scope_label(cluster_field_stored, cluster_turn_stored)}**"
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Было диалогов", n_before)
    with m2:
        st.metric("Удалено (точные)", dedup_exact)
    with m3:
        st.metric("Удалено (семант.)", dedup_sem)
    with m4:
        st.metric("Без дубликатов", len(stored.get("records") or []))
    with m5:
        st.metric(f"Кластеров {algorithm_name}", stored.get("n_clusters", 0))

    quality = stored.get("cluster_quality") or {}
    if quality:
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            st.metric("Шум, %", quality.get("noise_pct", 0))
        with q2:
            st.metric("Средний размер", quality.get("avg_cluster_size", 0))
        with q3:
            st.metric("Одиночных кластеров", quality.get("singleton_clusters", 0))
        with q4:
            st.metric("Кластеров ≤2", quality.get("small_clusters_le2", 0))

        n_records = len(stored.get("records") or [])
        n_clusters = int(stored.get("n_clusters") or 0)
        if n_records and n_clusters > max(5, n_records // 8):
            st.warning(
                "Много мелких кластеров — попробуйте увеличить **min_cluster_size** / "
                "**min_topic_size**, отключить **reduce_outliers** или включить **reduce_topics**."
            )
        if float(quality.get("noise_pct") or 0) > 35:
            st.info(
                "Большая доля шума — можно включить **reduce_outliers** или уменьшить "
                "**min_cluster_size** / **min_topic_size**."
            )

    dedup_details = stored.get("dedup_removed_details") or []
    if dedup_details:
        with st.expander(
            f"Удалённые дубликаты ({len(dedup_details)})",
            expanded=False,
        ):
            import pandas as pd

            df_dedup = pd.DataFrame(dedup_details)
            st.dataframe(df_dedup, use_container_width=True, hide_index=True)

    out_f = stored.get("output_field", "theme")
    non_empty = sum(
        1
        for r in stored.get("records") or []
        if str(r.get(out_f) or "").strip()
    )
    st.caption(f"С непустым **{out_f}**: {non_empty}")

    summary = cluster_summary_rows(
        stored.get("records") or [],
        cluster_field=stored.get("cluster_field", "dialog"),
        turn_index=int(stored.get("cluster_turn_index") or 0),
        output_field=stored.get("output_field", "cluster_label"),
        cluster_tfidf_words=parse_cluster_tfidf_words(stored.get("cluster_tfidf_words")),
    )
    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "TF-IDF": st.column_config.TextColumn(
                "TF-IDF",
                width="medium",
                help="Топ слов TF-IDF / c-TF-IDF для кластера.",
            ),
            "примеры": st.column_config.TextColumn(
                "примеры",
                width="large",
                help="До 3 полных диалогов из кластера.",
            ),
        },
    )

    render_cluster_visualizations(stored, key_prefix=key_prefix)
    render_similar_dialogs_block(
        stored,
        key_prefix=key_prefix,
        query_label=similar_query_label,
    )
    if show_balanced_sample:
        render_balanced_sample_block(stored, key_prefix=key_prefix)

    stats_report = stored.get("stats_report") or ""
    if stats_report:
        with st.expander("Подробная статистика", expanded=False):
            st.text(stats_report)

    records = stored.get("records") or []
    cluster_size_rows = cluster_size_chart_rows(records, output_field=out_f)
    cluster_ids = [int(row["cluster_id"]) for row in cluster_size_rows]
    cluster_label_by_id = {int(row["cluster_id"]): str(row["кластер"]) for row in cluster_size_rows}

    bal_stored_export = (
        st.session_state.get(f"{key_prefix}_balanced") if show_balanced_sample else None
    )
    has_balanced_export = bool(
        show_balanced_sample
        and bal_stored_export
        and bal_stored_export.get("records")
    )

    st.markdown("#### Экспорт выборки")
    st.caption("Сохранение в JSONL или в базу (Postgres + MinIO).")

    export_source_key = f"{key_prefix}_export_source"
    if has_balanced_export:
        export_source = st.radio(
            "Что экспортировать",
            options=["clusters", "balanced"],
            format_func=lambda x: (
                "Выбранные кластеры" if x == "clusters" else "Сбалансированная выборка"
            ),
            horizontal=True,
            key=export_source_key,
        )
    else:
        export_source = "clusters"

    if export_source == "balanced":
        export_records = list(bal_stored_export.get("records") or [])
        st.info(
            f"К сохранению: **{len(export_records)}** диалогов "
            f"(сбалансированная выборка из **{bal_stored_export.get('groups_used', 0)}** групп)."
        )
        subset_stats = ""
        out_name = "balanced_cluster_sample.jsonl"
        save_desc = f"Сбалансированная выборка ({algorithm_name})"
        export_disabled = not export_records
        name_placeholder = "balanced-sample-v1"
    else:
        pick_key = f"{key_prefix}_export_cluster_pick"
        if pick_key not in st.session_state:
            st.session_state[pick_key] = list(cluster_ids)
        else:
            pruned = [cid for cid in st.session_state[pick_key] if cid in cluster_ids]
            if not pruned and cluster_ids:
                st.session_state[pick_key] = list(cluster_ids)
            elif pruned != list(st.session_state[pick_key]):
                st.session_state[pick_key] = pruned

        ec1, ec2, ec3 = st.columns([2, 1, 1])
        with ec1:
            selected_cluster_ids = st.multiselect(
                "Кластеры для сохранения",
                options=cluster_ids,
                format_func=lambda cid: cluster_label_by_id.get(cid, str(cid)),
                key=pick_key,
            )
        with ec2:
            if st.button("Все кластеры", key=f"{key_prefix}_export_clusters_all"):
                st.session_state[pick_key] = list(cluster_ids)
                st.rerun()
        with ec3:
            if st.button("Снять все", key=f"{key_prefix}_export_clusters_none"):
                st.session_state[pick_key] = []
                st.rerun()

        selected_set = {int(cid) for cid in selected_cluster_ids}
        export_records = [
            r for r in records if int(r.get("cluster_id", -2)) in selected_set
        ]

        if selected_cluster_ids:
            st.info(
                f"К сохранению: **{len(export_records)}** диалогов "
                f"из **{len(selected_cluster_ids)}** кластер(ов)."
            )
        else:
            st.warning("Выберите хотя бы один кластер для экспорта.")

        subset_stats = stats_report
        if selected_cluster_ids and len(export_records) != len(records):
            picked_labels = ", ".join(
                cluster_label_by_id.get(cid, str(cid)) for cid in selected_cluster_ids
            )
            subset_stats = (
                f"Экспорт подмножества ({len(export_records)} из {len(records)} диалогов).\n"
                f"Кластеры: {picked_labels}\n\n{stats_report}"
            ).strip()

        out_name = f"{download_prefix}_{stored.get('cluster_field', 'dialog')}.jsonl"
        if selected_cluster_ids and len(selected_cluster_ids) < len(cluster_ids):
            out_name = f"{download_prefix}_subset_{stored.get('cluster_field', 'dialog')}.jsonl"

        save_desc = f"Выборка кластеров ({algorithm_name})"
        if selected_cluster_ids and len(selected_cluster_ids) < len(cluster_ids):
            save_desc += ": " + ", ".join(
                cluster_label_by_id.get(cid, str(cid)) for cid in selected_cluster_ids
            )
        export_disabled = not selected_cluster_ids
        name_placeholder = f"{download_prefix}-subset-v1"

    export_data = export_jsonl_with_stats(export_records, subset_stats)

    render_export_jsonl_actions(
        export_data,
        key_prefix=f"{key_prefix}_export",
        case_count=len(export_records),
        file_name=out_name,
        description=save_desc,
        disabled=export_disabled,
        name_placeholder=name_placeholder,
    )
    extra_fields = ""
    if algorithm_name in (CLUSTERING_ALGORITHM_LABEL, "BERTopic"):
        extra_fields = " · **topic_keywords** (ключевые слова c-TF-IDF)"
    if export_source == "balanced":
        st.caption(
            f"В каждой строке добавлены **cluster_id** и **{stored.get('output_field', 'cluster_label')}**"
            f"{extra_fields}. "
            "Сохраняются только диалоги, без #-отчёта в конце файла."
        )
    else:
        st.caption(
            f"В каждой строке добавлены **cluster_id** и **{stored.get('output_field', 'cluster_label')}**"
            f"{extra_fields}. "
            "В конец файла добавлена статистика в #-строках."
        )


def render_turn_scope_selector(
    label: str,
    *,
    field: str,
    key: str,
    disabled: bool = False,
) -> int:
    """0 — все фразы; 1 — первая; 2 — вторая и т.д."""
    if not is_history_cluster_field(field):
        return 0
    turn_options = [0, 1, 2, 3, 4, 5]
    return int(
        st.selectbox(
            label,
            options=turn_options,
            format_func=lambda n: "Все фразы" if n == 0 else f"Только {n}-я фраза",
            key=key,
            disabled=disabled,
            help=(
                "Для «Диалог» — номер реплики в history по порядку. "
                "Для «Фразы ассистента/пользователя» — номер среди реплик выбранной роли."
            ),
        )
    )


def record_field_selector(
    label: str,
    *,
    select_key: str,
    custom_key: str,
    disabled: bool = False,
) -> str:
    options = list(STANDARD_RECORD_FIELD_OPTIONS) + [CUSTOM_RECORD_FIELD_SENTINEL]
    choice = st.selectbox(
        label,
        options=options,
        format_func=lambda k: RECORD_FIELD_SELECT_LABELS[k],
        key=select_key,
        disabled=disabled,
    )
    custom_value = ""
    if choice == CUSTOM_RECORD_FIELD_SENTINEL:
        custom_value = st.text_input(
            "Ключ поля в JSONL",
            key=custom_key,
            disabled=disabled,
            placeholder="например subtopic",
            help="Scalar-поле на верхнем уровне записи (рядом с history).",
        )
    return resolve_record_field(choice, custom_value)


def render_nested_umap_settings(key_prefix: str) -> UmapSettings:
    with st.expander("UMAP", expanded=False):
        umap_cols1 = st.columns(4)
        with umap_cols1[0]:
            use_umap = st.checkbox(
                "Использовать UMAP",
                value=True,
                key=f"{key_prefix}_use_umap",
            )
        with umap_cols1[1]:
            umap_n_components = st.number_input(
                "n_components (0 = авто)",
                min_value=0,
                max_value=200,
                value=0,
                key=f"{key_prefix}_umap_n_components",
                disabled=not use_umap,
            )
        with umap_cols1[2]:
            umap_n_neighbors = st.number_input(
                "n_neighbors (0 = авто)",
                min_value=0,
                max_value=200,
                value=0,
                key=f"{key_prefix}_umap_n_neighbors",
                disabled=not use_umap,
            )
        with umap_cols1[3]:
            umap_init = st.selectbox(
                "init",
                options=["auto", "spectral", "random"],
                key=f"{key_prefix}_umap_init",
                disabled=not use_umap,
            )

        umap_cols2 = st.columns(4)
        with umap_cols2[0]:
            umap_min_dist = st.number_input(
                "min_dist",
                min_value=0.0,
                max_value=1.0,
                value=0.001,
                format="%.4f",
                key=f"{key_prefix}_umap_min_dist",
                disabled=not use_umap,
            )
        with umap_cols2[1]:
            umap_spread = st.number_input(
                "spread",
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                format="%.2f",
                key=f"{key_prefix}_umap_spread",
                disabled=not use_umap,
            )
        with umap_cols2[2]:
            umap_metric = st.selectbox(
                "metric",
                options=["cosine", "euclidean", "manhattan"],
                key=f"{key_prefix}_umap_metric",
                disabled=not use_umap,
            )
        with umap_cols2[3]:
            umap_random_state = st.number_input(
                "random_state",
                min_value=0,
                max_value=999999,
                value=42,
                key=f"{key_prefix}_umap_random_state",
                disabled=not use_umap,
            )

    return UmapSettings(
        enabled=use_umap,
        n_components=int(umap_n_components),
        n_neighbors=int(umap_n_neighbors),
        min_dist=float(umap_min_dist),
        spread=float(umap_spread),
        metric=umap_metric,
        random_state=int(umap_random_state),
        init=umap_init,
    )


def render_nested_hdbscan_settings(
    key_prefix: str,
    *,
    min_size_label: str,
) -> dict:
    with st.expander("HDBSCAN", expanded=False):
        hdb_cols1 = st.columns(4)
        with hdb_cols1[0]:
            min_size = st.number_input(
                min_size_label,
                min_value=2,
                max_value=500,
                value=5,
                key=f"{key_prefix}_min_size",
            )
        with hdb_cols1[1]:
            min_samples = st.number_input(
                "min_samples",
                min_value=1,
                max_value=100,
                value=1,
                key=f"{key_prefix}_min_samples",
            )
        with hdb_cols1[2]:
            hdb_metric = st.selectbox(
                "metric",
                options=["euclidean", "manhattan", "chebyshev"],
                key=f"{key_prefix}_hdb_metric",
            )
        with hdb_cols1[3]:
            hdb_selection = st.selectbox(
                "cluster_selection_method",
                options=["eom", "leaf"],
                key=f"{key_prefix}_hdb_selection",
            )

        hdb_cols2 = st.columns(2)
        with hdb_cols2[0]:
            hdb_alpha = st.number_input(
                "alpha",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                format="%.2f",
                key=f"{key_prefix}_hdb_alpha",
            )
        with hdb_cols2[1]:
            hdb_auto_scale = st.checkbox(
                "Авто-подстройка min_size по размеру выборки",
                value=True,
                key=f"{key_prefix}_hdb_auto_scale",
            )

    return {
        "min_size": int(min_size),
        "min_samples": int(min_samples),
        "metric": hdb_metric,
        "cluster_selection_method": hdb_selection,
        "alpha": float(hdb_alpha),
        "auto_scale": bool(hdb_auto_scale),
    }


def render_nested_bertopic_post_settings(key_prefix: str) -> dict:
    use_ctfidf = bool(st.session_state.get(f"{key_prefix}_use_ctfidf", True))
    with st.expander("Post-processing", expanded=False):
        post_cols = st.columns(4)
        with post_cols[0]:
            reduce_outliers = st.checkbox(
                "reduce_outliers",
                value=False,
                key=f"{key_prefix}_reduce_outliers",
            )
        with post_cols[1]:
            outlier_strategy = st.selectbox(
                "outlier strategy",
                options=["embeddings", "c-tf-idf", "distributions"],
                key=f"{key_prefix}_outlier_strategy",
                disabled=not reduce_outliers,
            )
        with post_cols[2]:
            reduce_topics = st.checkbox(
                "reduce_topics",
                value=False,
                key=f"{key_prefix}_reduce_topics",
            )
        with post_cols[3]:
            bertopic_top_n_words = st.number_input(
                "top_n_words (c-TF-IDF)",
                min_value=3,
                max_value=15,
                value=5,
                key=f"{key_prefix}_bertopic_top_n_words",
                disabled=not use_ctfidf,
            )

        nr_topics = "auto"
        if reduce_topics:
            nr_topics_mode = st.radio(
                "Число тем после слияния",
                ["auto", "фиксированное"],
                horizontal=True,
                key=f"{key_prefix}_nr_topics_mode",
            )
            if nr_topics_mode == "фиксированное":
                nr_topics = str(
                    st.number_input(
                        "nr_topics",
                        min_value=2,
                        max_value=200,
                        value=10,
                        key=f"{key_prefix}_nr_topics_fixed",
                    )
                )

    return {
        "reduce_outliers": reduce_outliers,
        "outlier_strategy": outlier_strategy,
        "reduce_topics": reduce_topics,
        "nr_topics": nr_topics,
        "bertopic_top_n_words": int(bertopic_top_n_words),
    }


def render_nested_tfidf_settings(key_prefix: str, *, bertopic: bool = False) -> dict:
    with st.expander("TF-IDF", expanded=False):
        use_ctfidf = True
        if bertopic:
            use_ctfidf = st.checkbox(
                "Использовать c-TF-IDF (ключевые слова тем)",
                value=True,
                key=f"{key_prefix}_use_ctfidf",
                help=(
                    "Если выключено — кластеризация только по эмбеддингам (HDBSCAN), "
                    "без словаря тем; подписи кластеров: «Тема 0», «Тема 1», …"
                ),
            )
        tfidf_top_n = st.number_input(
            "Топ слов в сводке и подсветке",
            min_value=3,
            max_value=30,
            value=int(st.session_state.get(f"{key_prefix}_tfidf_top_n", 10)),
            key=f"{key_prefix}_tfidf_top_n",
            disabled=bertopic and not use_ctfidf,
            help=(
                "Сколько ключевых слов показывать в таблице тем и подсветке примеров."
                if bertopic
                else None
            ),
        )
    return {
        "tfidf_top_n": int(tfidf_top_n),
        "use_ctfidf": bool(use_ctfidf),
    }


def render_clustering_pipeline_ui(
    key_prefix: str,
    *,
    bertopic: bool = False,
    supported_models: Optional[Sequence[str]] = None,
    fixed_cluster_field: Optional[str] = None,
    fixed_cluster_turn_index: Optional[int] = None,
) -> dict:
    with st.expander("1. Удалить дубликаты", expanded=False):
        dedup_cols = st.columns([2, 2, 2])
        with dedup_cols[0]:
            dedup_enabled = st.checkbox(
                "Удалить дубликаты перед кластеризацией",
                value=True,
                key=f"{key_prefix}_dedup_enabled",
            )
        with dedup_cols[1]:
            dedup_field = record_field_selector(
                "Поле для обнаружения дубликатов",
                select_key=f"{key_prefix}_dedup_field",
                custom_key=f"{key_prefix}_dedup_field_custom",
                disabled=not dedup_enabled,
            )
        with dedup_cols[2]:
            dedup_threshold = st.slider(
                "Порог cosine similarity",
                min_value=0.80,
                max_value=0.99,
                value=0.95,
                step=0.01,
                key=f"{key_prefix}_dedup_threshold",
                disabled=not dedup_enabled,
            )
        dedup_turn_index = render_turn_scope_selector(
            "Какую фразу сравнивать",
            field=dedup_field,
            key=f"{key_prefix}_dedup_turn",
            disabled=not dedup_enabled,
        )
        st.caption(
            "Точное совпадение нормализованного текста выбранного поля удаляется всегда, "
            "если удаление дубликатов включено."
        )

    output_label = "Куда записать название темы" if bertopic else "Куда записать название кластера"
    llm_model_key = f"{key_prefix}_llm_model"
    llm_sample_key = f"{key_prefix}_llm_sample"
    llm_prompt_key = f"{key_prefix}_llm_prompt"

    if supported_models is None:
        supported_models = _default_supported_models()

    with st.expander("2. Кластеризация", expanded=False):
        main_cols = st.columns([2, 2, 2])
        with main_cols[0]:
            if fixed_cluster_field:
                cluster_field = fixed_cluster_field
                st.text_input(
                    "Поле для кластеризации",
                    value=cluster_field,
                    disabled=True,
                    key=f"{key_prefix}_field_fixed",
                    help="Поле задано выше на странице результатов.",
                )
            else:
                cluster_field = record_field_selector(
                    "Поле для кластеризации",
                    select_key=f"{key_prefix}_field",
                    custom_key=f"{key_prefix}_field_custom",
                )
        with main_cols[1]:
            output_field = st.selectbox(
                output_label,
                options=["cluster_label", "theme", "cluster_reason"],
                key=f"{key_prefix}_output_field",
            )
        with main_cols[2]:
            with_llm = st.checkbox(
                "Использовать LLM для названия кластеров",
                value=False,
                key=f"{key_prefix}_with_llm",
            )

        if fixed_cluster_turn_index is not None:
            cluster_turn_index = int(fixed_cluster_turn_index)
            if is_history_cluster_field(cluster_field.split(".", 1)[0]):
                st.caption(
                    f"Фраза для кластеризации: **{turn_scope_label(cluster_turn_index)}** "
                    "(задано выше на странице результатов)."
                )
        else:
            cluster_turn_index = render_turn_scope_selector(
                "Какую фразу кластеризовать",
                field=cluster_field,
                key=f"{key_prefix}_cluster_turn",
            )

        if with_llm:
            llm_model_default = st.session_state.get("judge_evaluator_model", supported_models[0])
            llm_cols = st.columns([2, 1])
            with llm_cols[0]:
                model_idx = (
                    supported_models.index(llm_model_default)
                    if llm_model_default in supported_models
                    else 0
                )
                st.selectbox(
                    "Модель для названий",
                    options=supported_models,
                    index=min(model_idx, len(supported_models) - 1),
                    key=llm_model_key,
                )
            with llm_cols[1]:
                st.number_input(
                    "Диалогов в промпт",
                    min_value=1,
                    max_value=50,
                    value=25,
                    key=llm_sample_key,
                )
            if llm_prompt_key not in st.session_state:
                st.session_state[llm_prompt_key] = DEFAULT_CLUSTER_LLM_PROMPT_TEMPLATE
            with st.expander("Промпт LLM", expanded=False):
                st.text_area(
                    "Шаблон промпта",
                    key=llm_prompt_key,
                    height=300,
                    help=(
                        "Переменные: {cluster_label}, {n_records}, {field_label}, "
                        "{unique_block}, {formatted_sample}"
                    ),
                )

        umap_settings = render_nested_umap_settings(key_prefix)
        hdbscan_ui = render_nested_hdbscan_settings(
            key_prefix,
            min_size_label="min_topic_size" if bertopic else "min_cluster_size",
        )
        tfidf_ui = render_nested_tfidf_settings(key_prefix, bertopic=bertopic)
        tfidf_top_n = tfidf_ui["tfidf_top_n"]
        post_ui: dict = {}
        if bertopic:
            post_ui = render_nested_bertopic_post_settings(key_prefix)
            post_ui["use_ctfidf"] = tfidf_ui["use_ctfidf"]

    llm_model = st.session_state.get(
        llm_model_key,
        st.session_state.get("judge_evaluator_model", supported_models[0]),
    )
    llm_sample = int(st.session_state.get(llm_sample_key, 25))
    with_llm = bool(st.session_state.get(f"{key_prefix}_with_llm", False))
    llm_prompt_template = (
        st.session_state.get(llm_prompt_key, DEFAULT_CLUSTER_LLM_PROMPT_TEMPLATE)
        if with_llm
        else DEFAULT_CLUSTER_LLM_PROMPT_TEMPLATE
    )

    if bertopic:
        bertopic_settings = BertopicSettings(
            min_topic_size=hdbscan_ui["min_size"],
            min_samples=hdbscan_ui["min_samples"],
            hdbscan_metric=hdbscan_ui["metric"],
            cluster_selection_method=hdbscan_ui["cluster_selection_method"],
            alpha=hdbscan_ui["alpha"],
            auto_scale_min_topic_size=hdbscan_ui["auto_scale"],
            reduce_outliers=post_ui.get("reduce_outliers", False),
            outlier_strategy=post_ui.get("outlier_strategy", "embeddings"),
            reduce_topics=post_ui.get("reduce_topics", False),
            nr_topics=post_ui.get("nr_topics", "auto"),
            top_n_words=post_ui.get("bertopic_top_n_words", 5),
            use_ctfidf=bool(post_ui.get("use_ctfidf", True)),
        )
        hdbscan_settings = None
    else:
        hdbscan_settings = HdbscanSettings(
            min_cluster_size=hdbscan_ui["min_size"],
            min_samples=hdbscan_ui["min_samples"],
            metric=hdbscan_ui["metric"],
            cluster_selection_method=hdbscan_ui["cluster_selection_method"],
            alpha=hdbscan_ui["alpha"],
            auto_scale_min_cluster_size=hdbscan_ui["auto_scale"],
        )
        bertopic_settings = None

    return {
        "dedup_enabled": dedup_enabled,
        "dedup_field": dedup_field,
        "dedup_threshold": dedup_threshold,
        "dedup_turn_index": dedup_turn_index,
        "cluster_field": cluster_field,
        "cluster_turn_index": cluster_turn_index,
        "output_field": output_field,
        "with_llm": with_llm,
        "llm_model": llm_model,
        "llm_sample": llm_sample,
        "llm_prompt_template": llm_prompt_template,
        "umap_settings": umap_settings,
        "hdbscan_settings": hdbscan_settings,
        "bertopic_settings": bertopic_settings,
        "tfidf_top_n": tfidf_top_n,
    }

def bertopic_result_from_run(result, cfg: dict, *, clust_bad: int = 0) -> dict:
    """Сериализовать результат run_dialog_bertopic_clustering для session_state."""
    result.bad_lines = clust_bad
    return {
        "records": result.records,
        "stats_report": result.stats_report,
        "n_clusters": result.n_clusters,
        "cluster_field": result.cluster_field,
        "cluster_turn_index": result.cluster_turn_index,
        "output_field": result.output_field,
        "theme_counts": dict(result.theme_counts),
        "viz_points": result.viz_points,
        "n_input_before_dedup": result.n_input_before_dedup,
        "dedup_exact_removed": result.dedup_exact_removed,
        "dedup_semantic_removed": result.dedup_semantic_removed,
        "dedup_removed_details": result.dedup_removed_details,
        "dedup_field": result.dedup_field,
        "dedup_turn_index": result.dedup_turn_index,
        "algorithm": CLUSTERING_ALGORITHM_LABEL,
        "cluster_tfidf_words": serialize_cluster_tfidf_words(result.cluster_tfidf_words),
        "record_embeddings": result.record_embeddings,
        "cluster_quality": result.cluster_quality,
        "tfidf_top_n": cfg["tfidf_top_n"],
    }


def render_bertopic_clustering_block(
    records: list[dict],
    *,
    key_prefix: str,
    result_session_key: str,
    fingerprint_session_key: str,
    input_fingerprint: str,
    llm_api_key: str = "",
    llm_api_base: str = "",
    supported_models: Optional[Sequence[str]] = None,
    length_filter_expanded: bool = False,
    clust_bad: int = 0,
) -> None:
    """Кластеризация по готовому списку записей (без отдельной загрузки файла)."""
    if not records:
        st.info("Нет диалогов для кластеризации — включите хотя бы один диалог в выборку.")
        return

    stored = st.session_state.get(result_session_key)
    stored_fp = st.session_state.get(fingerprint_session_key)
    if stored and stored_fp and stored_fp != input_fingerprint:
        st.warning(
            "Выборка изменилась после последней кластеризации. "
            "Перезапустите кластеризацию, чтобы обновить результаты."
        )

    st.caption(f"К кластеризации доступно **{len(records)}** диалогов из текущей выборки.")

    with st.expander("📊 Длина диалогов", expanded=length_filter_expanded):
        records_for_run, _min_t, _max_t = render_dialog_length_distribution_panel(
            records,
            key_prefix=f"{key_prefix}_len",
            title="",
            filter_caption=(
                "Учитывается число реплик в `history`. "
                "Фильтр применяется к шагам «удалить дубликаты» и «кластеризация». "
                "0 слева — без нижней границы, 0 справа — без верхней."
            ),
        )
    if not records_for_run:
        st.warning("После фильтра по длине не осталось диалогов. Ослабьте фильтр.")
        return

    if len(records_for_run) != len(records):
        st.caption(
            f"К кластеризации будет использовано **{len(records_for_run)}** "
            f"из **{len(records)}** диалогов."
        )

    cfg = render_clustering_pipeline_ui(
        key_prefix,
        bertopic=True,
        supported_models=supported_models,
    )

    if st.button("Запустить кластеризацию", type="primary", key=f"{key_prefix}_run"):
        progress = st.progress(0.0, text="Старт…")
        status = st.empty()

        def on_progress(msg: str, frac: float) -> None:
            progress.progress(min(1.0, max(0.0, frac)), text=msg)
            status.caption(msg)

        try:
            result = run_dialog_bertopic_clustering(
                records_for_run,
                cluster_field=cfg["cluster_field"],
                output_field=cfg["output_field"],
                umap_settings=cfg["umap_settings"],
                bertopic_settings=cfg["bertopic_settings"],
                with_llm=cfg["with_llm"],
                llm_model=cfg["llm_model"] if cfg["with_llm"] else "",
                llm_api_key=llm_api_key,
                llm_api_base=llm_api_base,
                llm_sample_dialogs=int(cfg["llm_sample"]),
                llm_prompt_template=cfg["llm_prompt_template"],
                dedup_enabled=cfg["dedup_enabled"],
                dedup_field=cfg["dedup_field"],
                dedup_turn_index=cfg["dedup_turn_index"],
                cluster_turn_index=cfg["cluster_turn_index"],
                dedup_similarity_threshold=float(cfg["dedup_threshold"]),
                tfidf_top_n=cfg["tfidf_top_n"],
                on_progress=on_progress,
            )
            st.session_state[result_session_key] = bertopic_result_from_run(
                result, cfg, clust_bad=clust_bad
            )
            st.session_state[fingerprint_session_key] = input_fingerprint
            progress.progress(1.0, text="Готово")
            dedup_msg = ""
            if cfg["dedup_enabled"] and (
                result.dedup_exact_removed or result.dedup_semantic_removed
            ):
                dedup_msg = (
                    f" · дубликаты: −{result.dedup_exact_removed} точных, "
                    f"−{result.dedup_semantic_removed} семантических"
                )
            status.success(
                f"Кластеров: **{result.n_clusters}** · "
                f"диалогов после удаления дубликатов: **{len(result.records)}**"
                f"{dedup_msg}"
            )
            st.rerun()
        except Exception as e:
            progress.empty()
            status.empty()
            st.error(f"Ошибка кластеризации: {e}")

    stored = st.session_state.get(result_session_key)
    if not stored:
        return
    if st.session_state.get(fingerprint_session_key) != input_fingerprint:
        return

    render_clustering_results_panel(
        stored,
        key_prefix=key_prefix,
        algorithm_name=CLUSTERING_ALGORITHM_LABEL,
        download_prefix="clustered",
    )
