import copy
import hashlib
import json
import os
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

from ui.clustering_ui import render_bertopic_clustering_block
from ui.sample_storage_ui import render_export_jsonl_actions, render_jsonl_dataset_source
from ui.sample_comparison_ui import ComparisonSourceOption, render_dual_source_comparison

st.set_page_config(page_title="Анализ и редактирование данных", layout="wide")
st.title("📊 Анализ и редактирование данных")

st.markdown(
    "Загрузите датасет из файла или **из базы**, изучите статистику, **сформируйте выборку** "
    "(исключите лишние диалоги или добавьте новые вручную) и при необходимости "
    "запустите **кластеризацию** для поиска тем. Это необязательный шаг."
)


def normalize_role(role: str) -> str:
    role = str(role).strip().lower()
    if role in ["пользователь", "user", "клиент", "caller"]:
        return "user"
    if role in ["ассистент", "assistant", "оператор", "agent"]:
        return "assistant"
    return "user"


def load_jsonl_from_text(text: str) -> list:
    dialogs = []
    for line in text.strip().split("\n"):
        if line.strip() and not line.strip().startswith("#"):
            d = json.loads(line)
            for turn in d.get("history", []):
                turn["role"] = normalize_role(turn["role"])
            d["id"] = str(uuid.uuid4())
            d["original_history_length"] = len(d.get("history", []))
            dialogs.append(d)
    return dialogs


def _set_loaded_dialogs(dialogs: list, *, msg: str) -> None:
    st.session_state["current_dialogs"] = dialogs
    st.session_state["analysis_baseline_dialogs"] = copy.deepcopy(dialogs)
    st.session_state.pop("uploaded_file_id", None)
    st.session_state.pop("analysis_bertopic_result", None)
    st.session_state.pop("analysis_bertopic_fingerprint", None)
    st.session_state["_analysis_loaded_msg"] = msg
    st.rerun()


def _median_int(values: list[int]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def _dialog_turn_count(dialog: dict) -> int:
    return len(dialog.get("history", []))


def _passes_replica_filter(n_turns: int, min_r: int, max_r: int) -> bool:
    if min_r > 0 and n_turns < min_r:
        return False
    if max_r > 0 and n_turns > max_r:
        return False
    return True


def _selection_fingerprint(dialogs: list[dict]) -> str:
    parts = []
    for d in dialogs:
        did = d["id"]
        include = st.session_state.get(f"include_{did}", True)
        goal = st.session_state.get(
            f"edit_goal_{did}",
            d["goals"][0] if d.get("goals") else "Без цели",
        )
        max_len = max(1, d.get("original_history_length", len(d.get("history", []))))
        num = int(st.session_state.get(f"num_{did}", max_len))
        parts.append(f"{did}:{include}:{goal}:{num}")
    payload = "|".join(sorted(parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _records_for_clustering(dialogs: list[dict]) -> list[dict]:
    """Диалоги с галочкой «Включить», с обрезкой history по num_*."""
    out = []
    for dialog in dialogs:
        did = dialog["id"]
        if not st.session_state.get(f"include_{did}", True):
            continue
        max_len = max(1, dialog.get("original_history_length", len(dialog.get("history", []))))
        actual_len = int(st.session_state.get(f"num_{did}", max_len))
        actual_len = max(1, min(actual_len, max_len))
        rec = dict(dialog)
        rec["history"] = dialog["history"][:actual_len]
        goal = st.session_state.get(
            f"edit_goal_{did}",
            dialog["goals"][0] if dialog.get("goals") else "Без цели",
        )
        rec["goals"] = [goal] if goal != "Без цели" else []
        out.append(rec)
    return out


def _export_records(dialogs: list[dict], min_r: int, max_r: int) -> tuple[list[dict], int]:
    updated = []
    skipped = 0
    for dialog in dialogs:
        did = dialog["id"]
        if not st.session_state.get(f"include_{did}", True):
            continue
        n_full = len(dialog.get("history", []))
        if not _passes_replica_filter(n_full, min_r, max_r):
            skipped += 1
            continue
        max_len = max(1, dialog.get("original_history_length", n_full))
        actual_len = int(st.session_state.get(f"num_{did}", max_len))
        actual_len = max(1, min(actual_len, max_len))
        goal = st.session_state.get(
            f"edit_goal_{did}",
            dialog["goals"][0] if dialog.get("goals") else "Без цели",
        )
        out = dict(dialog)
        out["goals"] = [goal] if goal != "Без цели" else []
        out["history"] = dialog["history"][:actual_len]
        updated.append(out)
    return updated, skipped


def _render_data_comparison_tab(dialogs: list[dict]) -> None:
    """Вкладка «Сравнение данных»."""
    st.subheader("📈 Сравнение данных")
    st.caption(
        "Сравните две выборки: пресеты с этой страницы или любые JSONL "
        "(файл / сохранённые в БД). Метрики: баланс, покрытие, разнообразие."
    )

    min_r = int(st.session_state.get("reannotate_filter_min_turns", 0))
    max_r = int(st.session_state.get("reannotate_filter_max_turns", 0))
    export_dialogs, _ = _export_records(dialogs, min_r, max_r)
    baseline_dialogs = st.session_state.get("analysis_baseline_dialogs") or dialogs

    presets: list[ComparisonSourceOption] = [
        ComparisonSourceOption(
            option_id="load_baseline",
            label="Исходная (при загрузке)",
            records=baseline_dialogs,
        ),
        ComparisonSourceOption(
            option_id="export",
            label="Текущая (экспорт)",
            records=export_dialogs,
        ),
    ]

    selection_fp = _selection_fingerprint(dialogs)
    bertopic_stored = st.session_state.get("analysis_bertopic_result")
    bertopic_fp = st.session_state.get("analysis_bertopic_fingerprint")
    if bertopic_stored and bertopic_fp == selection_fp:
        clustered_records = bertopic_stored.get("records") or []
        out_f = bertopic_stored.get("output_field", "cluster_label")
        embeddings = bertopic_stored.get("record_embeddings")
        if clustered_records:
            presets.append(
                ComparisonSourceOption(
                    option_id="clustered",
                    label="После кластеризации",
                    records=clustered_records,
                    record_embeddings=embeddings,
                    embedding_source_records=clustered_records,
                    output_field=out_f,
                )
            )
        bal_stored = st.session_state.get("analysis_bertopic_balanced")
        if bal_stored and bal_stored.get("records"):
            presets.append(
                ComparisonSourceOption(
                    option_id="balanced",
                    label="Сбалансированная",
                    records=bal_stored["records"],
                    record_embeddings=embeddings,
                    embedding_source_records=clustered_records,
                    output_field=out_f,
                )
            )

    render_dual_source_comparison(
        presets,
        key_prefix="analysis_cmp",
        default_a_id="load_baseline",
        default_b_id="export",
    )


if "current_dialogs" not in st.session_state:
    st.session_state["current_dialogs"] = []

if "add_expanded" not in st.session_state:
    st.session_state["add_expanded"] = False

# --- Загрузка ---
st.subheader("📁 Загрузка данных")
_loaded_msg = st.session_state.pop("_analysis_loaded_msg", None)
if _loaded_msg:
    st.success(_loaded_msg)


def _on_analysis_dataset_text(text: str) -> None:
    try:
        dialogs = load_jsonl_from_text(text)
        if not dialogs:
            st.error("В файле не найдено диалогов.")
            return
        sample_name = st.session_state.pop("_analysis_loaded_sample_name", "")
        prefix = f"«{sample_name}» · " if sample_name else ""
        _set_loaded_dialogs(dialogs, msg=f"{prefix}загружено **{len(dialogs)}** диалогов.")
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")


def _on_analysis_dataset_db(raw: bytes, row: dict) -> None:
    st.session_state["_analysis_loaded_sample_name"] = row.get("name") or ""
    _on_analysis_dataset_text(raw.decode("utf-8", errors="replace"))


render_jsonl_dataset_source(
    key_prefix="analysis",
    on_text_loaded=_on_analysis_dataset_text,
    on_db_loaded=_on_analysis_dataset_db,
    upload_label="Загрузите файл .jsonl",
    file_types=["jsonl"],
)

dialogs = st.session_state["current_dialogs"]

all_goals = set()
for d in dialogs:
    if d.get("goals"):
        all_goals.add(d["goals"][0])
all_goals = sorted(all_goals) or ["Консультация"]

if not dialogs:
    tab_analyze, tab_compare = st.tabs(
        ["📊 Анализ и редактирование", "📈 Сравнение данных"]
    )
    with tab_analyze:
        st.info("Нет диалогов. Загрузите файл или добавьте диалог вручную.")
    with tab_compare:
        _render_data_comparison_tab(dialogs)
else:
    tab_analyze, tab_compare = st.tabs(
        ["📊 Анализ и редактирование", "📈 Сравнение данных"]
    )

    with tab_analyze:
        st.subheader("📈 Статистика")
        turn_counts = [_dialog_turn_count(d) for d in dialogs]
        total_turns = sum(turn_counts)
        tc_min = min(turn_counts)
        tc_max = max(turn_counts)
        tc_mean = total_turns / len(dialogs)
        tc_med = _median_int(turn_counts)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Всего диалогов", len(dialogs))
        with col2:
            st.metric("Всего реплик", total_turns)
        with col3:
            st.metric(
                "На диалог: мин / ср / медиана / макс",
                f"{tc_min} / {tc_mean:.1f} / {tc_med:.1f} / {tc_max}",
            )

        st.markdown(
            "**Распределение:** сколько **диалогов** имеют то или иное **число реплик**."
        )
        _repl_hist = Counter(turn_counts)
        _rows = sorted(_repl_hist.items(), key=lambda x: x[0])
        dist_df = pd.DataFrame(_rows, columns=["Число реплик", "Диалогов"])
        st.bar_chart(dist_df.set_index("Число реплик"), use_container_width=True)

        goal_counts = defaultdict(int)
        for d in dialogs:
            goal = d["goals"][0] if d.get("goals") else "Без цели"
            goal_counts[goal] += 1
        st.write("**Распределение по целям:**")
        stats_df = [{"Цель": goal, "Диалогов": count} for goal, count in goal_counts.items()]
        st.dataframe(stats_df, use_container_width=True)

        st.markdown("**Фильтр при экспорте по числу реплик**")
        st.caption("«0» слева — без нижней границы, «0» справа — без верхней.")
        _f1, _f2 = st.columns(2)
        with _f1:
            st.number_input(
                "Не меньше реплик",
                min_value=0,
                value=0,
                step=1,
                key="reannotate_filter_min_turns",
            )
        with _f2:
            st.number_input(
                "Не больше реплик",
                min_value=0,
                value=0,
                step=1,
                key="reannotate_filter_max_turns",
            )

        min_r = int(st.session_state.get("reannotate_filter_min_turns", 0))
        max_r = int(st.session_state.get("reannotate_filter_max_turns", 0))
        export_preview, _skipped_preview = _export_records(dialogs, min_r, max_r)
        st.info(
            f"В выборку для экспорта попадает **{len(export_preview)}** из **{len(dialogs)}** "
            f"диалогов (с учётом галочки «Включить» и фильтра по длине)."
        )

        st.subheader("➕ Добавить диалог вручную")
        with st.expander("Нажмите, чтобы добавить", expanded=st.session_state["add_expanded"]):
            new_goal = st.text_input("Цель нового диалога", key="new_dialog_goal")
            new_history = st.text_area(
                "История диалога (формат: роль | текст)",
                height=150,
                key="new_dialog_history",
            )
            if st.button("Добавить диалог", key="add_new_dialog_btn"):
                if not new_goal.strip():
                    st.warning("Укажите цель диалога.")
                elif not new_history.strip():
                    st.warning("Укажите историю диалога.")
                else:
                    history = []
                    valid = True
                    for line in new_history.strip().split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        if "|" not in line:
                            st.warning(f"Некорректная строка (отсутствует '|'): '{line}'")
                            valid = False
                            break
                        role_part, content_part = line.split("|", 1)
                        role = normalize_role(role_part.strip())
                        content = content_part.strip()
                        if content:
                            history.append({"role": role, "content": content})

                    if valid and history:
                        new_dialog = {
                            "goals": [new_goal.strip()],
                            "history": history,
                            "original_history_length": len(history),
                            "id": str(uuid.uuid4()),
                        }
                        st.session_state["current_dialogs"].insert(0, new_dialog)
                        st.session_state.pop("analysis_bertopic_result", None)
                        st.session_state.pop("analysis_bertopic_fingerprint", None)
                        st.success(f"✅ Добавлен диалог с целью: **{new_goal.strip()}**")
                        st.session_state["add_expanded"] = True
                        st.rerun()
                    elif valid and not history:
                        st.warning("Не удалось извлечь ни одной реплики.")

        goal_options_set = {d["goals"][0] for d in dialogs if d.get("goals")}
        goal_options = sorted(goal_options_set) + ["Без цели"]
        if not goal_options:
            goal_options = ["Без цели"]

        with st.expander("✏️ Переразметка", expanded=False):
            st.caption(
                "Снимите галочку «Включить», чтобы убрать диалог из выборки. "
                "Выборка используется при экспорте и при опциональной кластеризации."
            )

            dialogs_per_page = 10
            total_dialogs_n = len(dialogs)
            total_pages = max(1, (total_dialogs_n + dialogs_per_page - 1) // dialogs_per_page)

            if "analysis_current_page" not in st.session_state:
                st.session_state.analysis_current_page = 0
            if "reannotate_current_page" in st.session_state:
                st.session_state.analysis_current_page = st.session_state.reannotate_current_page

            if st.session_state.analysis_current_page >= total_pages:
                st.session_state.analysis_current_page = max(0, total_pages - 1)

            current_page = st.session_state.analysis_current_page
            pc1, pc2, pc3 = st.columns([1, 2, 1])
            with pc1:
                if st.button("← Назад", disabled=(current_page == 0), key="analysis_page_prev"):
                    st.session_state.analysis_current_page -= 1
                    st.rerun()
            with pc2:
                included_n = sum(
                    1 for d in dialogs if st.session_state.get(f"include_{d['id']}", True)
                )
                st.markdown(
                    f"**Страница {current_page + 1} из {total_pages}** · "
                    f"всего **{total_dialogs_n}** · включено **{included_n}** · "
                    f"по **{dialogs_per_page}** на страницу"
                )
            with pc3:
                if st.button(
                    "Вперёд →",
                    disabled=(current_page >= total_pages - 1),
                    key="analysis_page_next",
                ):
                    st.session_state.analysis_current_page += 1
                    st.rerun()

            start_idx = current_page * dialogs_per_page
            end_idx = min(start_idx + dialogs_per_page, total_dialogs_n)
            page_dialogs = dialogs[start_idx:end_idx]

            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                if st.button("Выбрать все", key="select_all"):
                    for d in dialogs:
                        st.session_state[f"include_{d['id']}"] = True
                    st.rerun()
            with bc2:
                if st.button("Снять все", key="deselect_all"):
                    for d in dialogs:
                        st.session_state[f"include_{d['id']}"] = False
                    st.rerun()
            with bc3:
                if st.button("Инвертировать", key="invert_all"):
                    for d in dialogs:
                        key = f"include_{d['id']}"
                        st.session_state[key] = not st.session_state.get(key, True)
                    st.rerun()

            for d in dialogs:
                dialog_id = d["id"]
                include_key = f"include_{dialog_id}"
                if include_key not in st.session_state:
                    st.session_state[include_key] = True

            for _dlg in dialogs:
                _gid = _dlg["id"]
                _gk = f"edit_goal_{_gid}"
                _nk = f"num_{_gid}"
                _mx = max(1, _dlg.get("original_history_length", len(_dlg.get("history", []))))
                if _gk not in st.session_state:
                    st.session_state[_gk] = (
                        _dlg["goals"][0] if _dlg.get("goals") else "Без цели"
                    )
                if _nk not in st.session_state:
                    st.session_state[_nk] = _mx
                _cv = st.session_state[_nk]
                if _cv < 1:
                    st.session_state[_nk] = 1
                elif _cv > _mx:
                    st.session_state[_nk] = _mx

            st.markdown("---")

            show_edit = st.checkbox(
                "Редактировать цель и число реплик",
                value=False,
                key="analysis_show_edit",
            )

            for dialog in page_dialogs:
                dialog_id = dialog["id"]
                include_key = f"include_{dialog_id}"
                goal_key = f"edit_goal_{dialog_id}"
                num_key = f"num_{dialog_id}"

                current_goal = st.session_state[goal_key]
                if current_goal not in goal_options:
                    goal_options = [current_goal] + goal_options

                max_len = dialog.get("original_history_length", len(dialog.get("history", [])))
                max_len = max(1, max_len)
                actual_len = st.session_state[num_key]
                displayed_history = dialog["history"][:actual_len]

                with st.container():
                    if show_edit:
                        col_check, col_goal, col_len, col_content = st.columns(
                            [0.5, 1.5, 0.8, 4]
                        )
                        with col_check:
                            st.checkbox("Включить", key=include_key)
                        with col_goal:
                            st.selectbox(
                                "Цель",
                                options=goal_options,
                                index=goal_options.index(current_goal),
                                key=goal_key,
                                label_visibility="collapsed",
                            )
                        with col_len:
                            st.number_input(
                                "Реплик",
                                min_value=1,
                                max_value=max_len,
                                key=num_key,
                                label_visibility="visible",
                            )
                        with col_content:
                            lines = []
                            for turn in displayed_history:
                                label = (
                                    "👤 Пользователь"
                                    if turn["role"] == "user"
                                    else "💼 Ассистент"
                                )
                                lines.append(f"{label}: {turn['content']}")
                            st.text("\n".join(lines))
                    else:
                        col_check, col_content = st.columns([0.6, 5])
                        with col_check:
                            st.checkbox("Включить", key=include_key)
                        with col_content:
                            goal_label = current_goal
                            lines = []
                            for turn in displayed_history:
                                label = (
                                    "👤 Пользователь"
                                    if turn["role"] == "user"
                                    else "💼 Ассистент"
                                )
                                lines.append(f"{label}: {turn['content']}")
                            preview = "\n".join(lines)
                            st.markdown(f"**{goal_label}** · {actual_len} репл.")
                            st.text(preview)

                st.markdown("---")

        min_r = int(st.session_state.get("reannotate_filter_min_turns", 0))
        max_r = int(st.session_state.get("reannotate_filter_max_turns", 0))
        updated_dialogs, skipped_by_turn_filter = _export_records(dialogs, min_r, max_r)

        st.subheader("📥 Экспорт выборки")
        _msg_save = f"Будет сохранено **{len(updated_dialogs)}** диалогов"
        if skipped_by_turn_filter:
            _msg_save += (
                f" · исключено фильтром по длине (включены галочкой): "
                f"**{skipped_by_turn_filter}**"
            )
        st.success(_msg_save)

        if updated_dialogs:
            jsonl_lines = [
                json.dumps(d, ensure_ascii=False, separators=(",", ":"))
                for d in updated_dialogs
            ]
            jsonl_content = "\n".join(jsonl_lines)
            render_export_jsonl_actions(
                jsonl_content + ("\n" if jsonl_content else ""),
                key_prefix="analysis_export",
                case_count=len(updated_dialogs),
                file_name="dataset_sample.jsonl",
                description="Выборка со страницы «Анализ и редактирование данных»",
                name_placeholder="analysis-sample-v1",
            )

        selection_fp = _selection_fingerprint(dialogs)
        cluster_records = _records_for_clustering(dialogs)

        st.subheader("🔬 Кластеризация")
        st.caption(
            "Необязательный шаг: группировка диалогов по темам. "
            "Используются только диалоги с галочкой «Включить»."
        )
        with st.expander("Настройки и запуск кластеризации", expanded=False):
            st.markdown(
                "Пайплайн: **1) удалить дубликаты** → "
                "**2) кластеризация** (эмбеддинги → UMAP → HDBSCAN → c-TF-IDF)."
            )
            render_bertopic_clustering_block(
                cluster_records,
                key_prefix="analysis_bertopic",
                result_session_key="analysis_bertopic_result",
                fingerprint_session_key="analysis_bertopic_fingerprint",
                input_fingerprint=selection_fp,
                llm_api_key=os.getenv("LITELLM_API_KEY", ""),
                llm_api_base=os.getenv("LITELLM_API_BASE", ""),
            )

    with tab_compare:
        _render_data_comparison_tab(dialogs)
