"""Общее отображение строк результатов бенчмарка (Streamlit)."""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import pandas as pd
import streamlit as st

from benchmarking.runner import (
    LLM_JUDGE_EVAL_MODE,
    benchmark_criterion_accuracy_summary,
    benchmark_mean_criterion_score,
    extract_json_object_from_llm_text,
    format_eval_field_for_display,
    is_llm_judge_eval_mode,
    is_semantic_similarity_eval_mode,
    llm_eval_scoring_field_names,
    parse_llm_eval_fields,
    row_criterion_accuracy,
    row_mean_criterion_score,
)
from benchmarking.bootstrap import (
    benchmark_accuracy_bootstrap,
    benchmark_criterion_bootstrap_cis,
    benchmark_mean_criterion_score_bootstrap,
    format_bootstrap_ci_caption,
    format_bootstrap_interval,
)
from common.security import redact_secrets_for_display
from common.utils import (
    ASSISTANT_ROLES_REPLY_TIMING,
    USER_ROLES_REPLY_TIMING,
    benchmark_mean_dialog_durations,
    benchmark_mean_reply_times,
    history_prepared_for_timing_metrics,
    metric_float,
    role_for_reply_timing,
    row_avg_reply_times_from_history,
    row_dialog_duration_sec_effective,
    turn_reply_duration_sec,
)

DIALOGS_PER_PAGE = 10


def render_criterion_accuracy_metrics(
    results: list,
    llm_eval_fields: str = "",
    *,
    key_prefix: str = "criterion_bootstrap",
) -> None:
    """Сводка точности по критериям LLM-судьи с bootstrap-ДИ (то же, что bootstrap-блок)."""
    render_benchmark_bootstrap_metrics(
        results,
        llm_eval_fields=llm_eval_fields,
        eval_mode=LLM_JUDGE_EVAL_MODE,
        key_prefix=key_prefix,
        show_overall_accuracy=False,
    )


def render_benchmark_bootstrap_metrics(
    results: list,
    llm_eval_fields: str = "",
    eval_mode: str = "",
    *,
    key_prefix: str = "bootstrap",
    show_overall_accuracy: bool = True,
    require_run_button: bool = True,
    data_fingerprint: str = "",
) -> None:
    """Bootstrap CI по перцентилям: общая точность и каждый критерий LLM-судьи."""
    rows = [r for r in results if isinstance(r, dict)]
    if not rows:
        return

    st.markdown("##### Bootstrap · доверительные интервалы")

    run_state_key = f"{key_prefix}_bootstrap_done"
    fp_key = f"{key_prefix}_bootstrap_fp"
    if data_fingerprint and st.session_state.get(fp_key) != data_fingerprint:
        st.session_state.pop(run_state_key, None)
        st.session_state[fp_key] = data_fingerprint

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        n_bootstrap = st.number_input(
            "Число bootstrap-выборок",
            min_value=500,
            max_value=50000,
            value=5000,
            step=500,
            key=f"{key_prefix}_n_bootstrap",
        )
    with c2:
        confidence_pct = st.slider(
            "Уровень доверия",
            min_value=80,
            max_value=99,
            value=95,
            step=1,
            format="%d%%",
            key=f"{key_prefix}_confidence",
        )
        confidence = confidence_pct / 100.0
    with c3:
        st.write("")
        st.write("")
        if st.button(
            "Рассчитать bootstrap",
            type="primary",
            key=f"{key_prefix}_bootstrap_btn",
        ):
            st.session_state[run_state_key] = True

    if require_run_button and not st.session_state.get(run_state_key):
        st.caption(
            "Нажмите **«Рассчитать bootstrap»**, чтобы построить доверительные интервалы "
            f"по **{len(rows)}** кейсам."
        )
        return

    if len(rows) < 2:
        st.info(
            "Для расчёта доверительных интервалов нужно минимум **2** кейса в выборке. "
            "Ниже — только точечные оценки без ДИ."
        )
        if show_overall_accuracy:
            avg = sum(float(r.get("accuracy", 0) or 0) for r in rows) / len(rows)
            st.markdown(f"**Общая точность:** {avg:.2%} · n={len(rows)}")
        if is_llm_judge_eval_mode(eval_mode):
            summary = benchmark_criterion_accuracy_summary(rows, llm_eval_fields)
            if summary:
                st.markdown("**По критериям LLM-судьи:**")
                for field, avg in summary.items():
                    st.markdown(f"- `{field}`: **{avg:.2%}**")
        return

    boot_kw = {
        "n_bootstrap": int(n_bootstrap),
        "confidence": float(confidence),
        "seed": 42,
    }
    conf_pct = int(round(float(confidence) * 100))

    if show_overall_accuracy:
        acc_ci = benchmark_accuracy_bootstrap(rows, **boot_kw)
        if acc_ci is not None:
            st.markdown(
                f"**Общая точность (mean по кейсам):** "
                f"{format_bootstrap_interval(acc_ci)} · n={acc_ci.n}, {conf_pct}% ДИ"
            )

    if is_llm_judge_eval_mode(eval_mode):
        scoring = llm_eval_scoring_field_names(
            parse_llm_eval_fields(llm_eval_fields)
        )
        crit_cis = benchmark_criterion_bootstrap_cis(
            rows, llm_eval_fields, **boot_kw
        )
        if crit_cis:
            st.markdown("**По критериям LLM-судьи:**")
            metric_items = list(crit_cis.items())
            if len(scoring) > 1:
                mean_ci = benchmark_mean_criterion_score_bootstrap(
                    rows, llm_eval_fields, **boot_kw
                )
                if mean_ci is not None:
                    metric_items = [("__mean__", mean_ci)] + metric_items
            cols = st.columns(min(len(metric_items), 4))
            for idx, (field, ci) in enumerate(metric_items):
                label = (
                    "Средний балл"
                    if field == "__mean__"
                    else f"`{field}`"
                )
                with cols[idx % len(cols)]:
                    st.metric(label, f"{ci.estimate:.2%}")
                    st.caption(format_bootstrap_ci_caption(ci, confidence=confidence))

    st.caption(
        "Percentile bootstrap: повторная выборка **кейсов** с возвращением; на каждой "
        "итерации пересчитывается метрика; границы ДИ — перцентили распределения."
    )


def render_timing_summary_metrics(
    results: list,
    *,
    show_help: bool = True,
) -> None:
    """
    Сводка времени по загруженным строкам результатов: метрики всегда видны,
    при отсутствии чисел в JSON — значение «—» и подсказка.
    """
    rows = [r for r in results if isinstance(r, dict)]
    if not rows:
        return

    m_as, m_us = benchmark_mean_reply_times(rows)
    md = benchmark_mean_dialog_durations(rows)
    br = next(
        (
            v
            for d in rows
            if (v := metric_float(d.get("benchmark_run_duration_sec"))) is not None
        ),
        None,
    )

    st.markdown("##### Время")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Ср. ответ ассистента",
            f"{m_as:.3f} с" if m_as is not None else "—",
        )
    with c2:
        st.metric(
            "Ср. ответ пользователя",
            f"{m_us:.3f} с" if m_us is not None else "—",
        )
    with c3:
        st.metric(
            "Ср. время диалога",
            f"{md:.3f} с" if md is not None else "—",
        )
    with c4:
        st.metric(
            "Прогон  ",
            f"{br:.3f} с" if br is not None else "—",
        )
    if show_help and m_as is None and m_us is None and md is None and br is None:
        st.caption(
            "В данных нет меток времени. Нужен **новый прогон** с обновлённым воркером: "
            "в `history` — `reply_duration_sec` на репликах, в строке — `dialog_duration_sec` и "
            "`benchmark_run_duration_sec`."
        )
    elif show_help and m_as is None and m_us is None and (md is not None or br is not None):
        st.caption(
            "Средние по репликам считаются из `reply_duration_sec` в `history`; без них — только время диалога и прогон."
        )


def _turn_duration_suffix(msg: dict) -> str:
    sec = turn_reply_duration_sec(msg) if isinstance(msg, dict) else None
    if sec is not None:
        return f" ({sec:.3f} с)"
    return ""


def format_goals_for_result_row(result: dict) -> str:
    """Цели для заголовка/строки: goals_text (после бенчмарка) или исходные goals из JSONL."""
    gt = result.get("goals_text")
    if isinstance(gt, str) and gt.strip():
        return gt.strip()
    g = result.get("goals")
    if isinstance(g, str) and g.strip():
        return g.strip()
    if isinstance(g, list) and g:
        return ", ".join(str(x) for x in g)
    return "Без цели"


def render_dialogs(
    dialogs: list,
    disp_eval_mode: str,
    llm_eval_fields_disp: str = "",
    *,
    start_index: int = 0,
    expander_row_label: str = "Диалог",
    expander_title_show_accuracy_icon: bool = True,
    show_dialog_level_api_response: bool = False,
) -> None:
    for idx, result in enumerate(dialogs):
        global_idx = start_index + idx
        icon = "✅" if result.get("accuracy", 0) == 1 else "❌"
        title_icon = f"{icon} " if expander_title_show_accuracy_icon else ""
        dialog_id = result.get("dialog_id", f"dlg_{global_idx}")
        scenario_id = result.get("scenario_id")
        goals = format_goals_for_result_row(result)
        sid_short = f" | сценарий: {scenario_id}" if scenario_id is not None else ""
        with st.expander(
            f"{title_icon}{expander_row_label} {global_idx + 1} (ID: {dialog_id}){sid_short}: {goals}",
            expanded=False,
        ):
            st.write(f"**🆔 ID диалога:** `{dialog_id}`")
            if scenario_id is not None:
                st.write(f"**📋 scenario_id:** `{scenario_id}`")
            st.write(f"**🎯 Цель:** {goals}")
            ctx = result.get("context")
            if isinstance(ctx, dict) and ctx:
                with st.expander("📎 context (из датасета)", expanded=False):
                    st.json(redact_secrets_for_display(ctx))
            elif isinstance(ctx, str) and ctx.strip():
                with st.expander("📎 context (строка из датасета)", expanded=False):
                    st.text(ctx)
            st.write(f"**🧠 Ответ ассистента:** {result.get('predicted_intents', '—')}")
            acc_ok = result.get("accuracy", 0) == 1
            acc_val = metric_float(result.get("accuracy"))
            repeats_k = int(result.get("repeats_per_case") or 1)
            if repeats_k > 1 and acc_val is not None:
                st.write(
                    f"**📊 Точность (mean@{repeats_k}):** {acc_val:.2%} "
                    f"· pass≥1: {'✅' if result.get('pass_at_least_once') else '❌'}"
                    f" · pass all: {'✅' if result.get('pass_all') else '❌'}"
                )
                ra = result.get("repeat_accuracies")
                if isinstance(ra, list) and ra:
                    st.caption(f"Повторы: `{ra}`")
            else:
                st.write(f"**📊 Точность (accuracy):** {'✅' if acc_ok else '❌'}")
            if result.get("dm_max_turns_exceeded"):
                st.error(
                    result.get("dm_max_turns_note")
                    or "Прервано по лимиту max_turns (вызовов form_next_phrase); диалог засчитан как неверный."
                )
                if result.get("dm_max_turns_limit") is not None:
                    st.caption(
                        f"Лимит max_turns в конфиге: **{result.get('dm_max_turns_limit')}** · "
                        f"вызовов form_next_phrase: **{result.get('dm_form_next_phrase_calls', '—')}**"
                    )
            if result.get("dm_scenario_api_timeout"):
                st.error(
                    result.get("dm_session_error")
                    or (
                        "Долгое выполнение: ответ API сценария (Dialog Manager) не получен за "
                        f"{result.get('dm_scenario_api_timeout_sec', 120)} с. "
                        "Диалог с ошибкой; ниже — накопленная история."
                    )
                )
            elif result.get("dm_empty_operator_reply"):
                st.error(
                    result.get("dm_session_error")
                    or (
                        "Сценарий не вернул озвучиваемую реплику оператора; следующая реплика "
                        "пользователя не отправлялась, пока нет ответа ассистента."
                    )
                )
            elif result.get("dm_session_error") and not result.get("dm_max_turns_exceeded"):
                st.warning(str(result.get("dm_session_error")))
            st.write(f"**🔢 Ходов:** {result.get('turns', '—')}")
            st.markdown("**Время (по этому диалогу)**")
            _dd = row_dialog_duration_sec_effective(result)
            _ra, _ru = row_avg_reply_times_from_history(result.get("history"))
            _a = metric_float(result.get("avg_assistant_reply_sec"))
            _u = metric_float(result.get("avg_user_reply_sec"))
            if _a is None:
                _a = _ra
            if _u is None:
                _u = _ru
            _br = metric_float(result.get("benchmark_run_duration_sec"))
            st.write(
                "**Время прогона диалога:** "
                + (f"{_dd:.3f} с" if _dd is not None else "—")
            )
            st.write(
                "**Среднее время ответа ассистента:** "
                + (f"{_a:.3f} с" if _a is not None else "—")
            )
            st.write(
                "**Среднее время ответа пользователя:** "
                + (f"{_u:.3f} с" if _u is not None else "—")
            )
            st.write(
                "**Время полного прогона бенчмарка  :** "
                + (f"{_br:.3f} с" if _br is not None else "—")
            )

            if is_llm_judge_eval_mode(disp_eval_mode):
                st.write(
                    f"**🤖 LLM-судья (итог):** {'✅ Да' if result.get('result') else '❌ Нет'}"
                )
                expected_fields_list = parse_llm_eval_fields(llm_eval_fields_disp)
                scoring_fields = llm_eval_scoring_field_names(expected_fields_list)
                per_crit = row_criterion_accuracy(result, scoring_fields)
                if per_crit:
                    st.write("**📊 Точность по критериям:**")
                    for field in scoring_fields:
                        if field not in per_crit:
                            continue
                        ok = per_crit[field] == 1.0
                        st.write(f"- `{field}`: {'✅' if ok else '❌'}")
                    if len(scoring_fields) > 1:
                        mean_row = row_mean_criterion_score(result, scoring_fields)
                        if mean_row is not None:
                            st.write(
                                f"**📈 Средний балл по критериям:** {mean_row:.2%}"
                            )
                extra_keys = set(expected_fields_list) - set(scoring_fields)
                parsed_fallback = extract_json_object_from_llm_text(
                    result.get("raw_output") or ""
                )
                if extra_keys:
                    st.write("**📝 Дополнительные поля из оценки:**")
                    for key in sorted(extra_keys):
                        val = result.get(key)
                        if val is None and parsed_fallback:
                            val = parsed_fallback.get(key)
                        st.write(f"- `{key}`: {format_eval_field_for_display(val)}")
                with st.expander(
                    "🔍 Полный ответ LLM-оценщика (сырой вывод)", expanded=False
                ):
                    _ro = (result.get("raw_output") or "").strip()
                    st.text(_ro if _ro else "[Нет данных]")
                case_repeats = result.get("case_repeats")
                if isinstance(case_repeats, list) and len(case_repeats) > 1:
                    with st.expander(
                        f"🔁 Детали повторов ({len(case_repeats)})", expanded=False
                    ):
                        for rep in case_repeats:
                            if not isinstance(rep, dict):
                                continue
                            ri = rep.get("repeat_index", "?")
                            ra = rep.get("accuracy")
                            st.markdown(
                                f"**Повтор {ri}:** accuracy={ra} · turns={rep.get('turns', '—')}"
                            )
                            if rep.get("dm_session_error") or rep.get(
                                "benchmark_run_exception"
                            ):
                                st.caption(
                                    str(
                                        rep.get("dm_session_error")
                                        or rep.get("benchmark_run_exception")
                                    )
                                )
            elif is_semantic_similarity_eval_mode(disp_eval_mode):
                sim = result.get("semantic_similarity")
                thr = result.get("semantic_similarity_threshold")
                st.write(
                    f"**🔗 Семантическое сходство:** "
                    f"{sim if sim is not None else '—'}"
                    + (f" (порог {thr})" if thr is not None else "")
                )
                st.write(
                    f"**Совпадение по порогу:** "
                    f"{'✅ да' if result.get('semantic_match') else '❌ нет'}"
                )
                st.write(
                    f"**Последний ответ ассистента:** {result.get('semantic_pred_text') or '—'}"
                )
                st.write(
                    f"**Эталон из разметки:** {result.get('semantic_ref_text') or '—'}"
                )
            elif disp_eval_mode == "Достигнут блок (block_id)":
                st.write(
                    f"**🧱 Цель по блоку:** "
                    f"{'✅ достигнута' if result.get('block_goal_met') else '❌ нет'}"
                )
                st.caption(
                    "**Точность (accuracy)** совпадает с «цель по блоку» (блок из JSONL/context/goals). "
                    "Ранняя остановка по полю «Остановиться после блока» не подменяет этот критерий."
                )
                dms = result.get("dm_stop_after_block_met")
                if dms is not None:
                    st.write(
                        f"**Порог «остановиться после блока» (сессия):** "
                        f"{'✅ достигнут' if dms else '❌ нет'}"
                    )
                st.write(
                    f"**target_block_ids:** `{result.get('target_block_ids', '—')}` · "
                    f"**visited_block_ids_dm:** `{result.get('visited_block_ids_dm', '—')}`"
                )
            else:
                st.write(
                    f"**🔍 Значение для оценки:** {result.get('eval_field_value', '—')}"
                )

            st.write("**📜 История диалога:**")
            for msg in history_prepared_for_timing_metrics(result.get("history")):
                role_norm = role_for_reply_timing(msg)
                if role_norm in USER_ROLES_REPLY_TIMING:
                    st.text(
                        f"👤 Пользователь: {msg.get('content', '')}{_turn_duration_suffix(msg)}"
                    )
                elif role_norm in ASSISTANT_ROLES_REPLY_TIMING:
                    raw = msg.get("content", "") or ""
                    c = raw.strip() if isinstance(raw, str) else ""
                    fr = msg.get("full_response")
                    if c:
                        line = raw if isinstance(raw, str) else str(raw)
                    elif isinstance(fr, dict) and fr.get("scenario_finished"):
                        line = "[сценарий завершён без озвучиваемой реплики]"
                    else:
                        line = "[ответ без озвучиваемого текста]"
                    st.text(f"💼 Ассистент: {line}{_turn_duration_suffix(msg)}")
                    if msg.get("is_external") and msg.get("full_response"):
                        with st.expander("📡 API Response (этот шаг)", expanded=False):
                            st.json(
                                redact_secrets_for_display(msg["full_response"])
                            )
                    elif msg.get("is_external"):
                        st.caption("📡 API Response: [нет данных]")

                    full_resp = msg.get("full_response", {})
                    if (
                        not msg.get("is_external")
                        and isinstance(full_resp, dict)
                        and full_resp.get("tool_calls")
                    ):
                        with st.expander(
                            "🔧 Вызовы инструментов (tool_calls)", expanded=False
                        ):
                            st.json(
                                redact_secrets_for_display(full_resp["tool_calls"])
                            )
                else:
                    content = msg.get("content", "") or ""
                    if msg.get("role") == "tool" or role_norm == "tool":
                        st.text(f"🔧 {content}{_turn_duration_suffix(msg)}")
                    else:
                        st.text(
                            f"🔧 {role_norm or msg.get('role')}: {content}{_turn_duration_suffix(msg)}"
                        )

            if show_dialog_level_api_response and result.get("mode") == "Внешний URL":
                fr = result.get("full_response")
                if fr:
                    with st.expander("📡 Полный ответ от API (raw)", expanded=False):
                        st.json(
                            redact_secrets_for_display(fr)
                            if isinstance(fr, dict)
                            else {}
                        )


def render_dialogs_paginated(
    dialogs: list,
    disp_eval_mode: str,
    llm_eval_fields_disp: str = "",
    *,
    session_page_key: str,
    list_entity_caption: str = "диалоги",
    expander_row_label: str = "Диалог",
    expander_title_show_accuracy_icon: bool = True,
    show_dialog_level_api_response: bool = False,
    show_timing_summary: bool = True,
) -> None:
    """Как на странице «Разметка»: по 10 диалогов на страницу, кнопки Назад / Вперёд."""
    n = len(dialogs)
    if n == 0:
        st.info("Нет записей для отображения.")
        return

    total_pages = max(1, (n + DIALOGS_PER_PAGE - 1) // DIALOGS_PER_PAGE)
    if session_page_key not in st.session_state:
        st.session_state[session_page_key] = 0
    try:
        cp_raw = st.session_state[session_page_key]
        cp = max(0, min(int(cp_raw), total_pages - 1))
    except (TypeError, ValueError):
        cp = 0
    st.session_state[session_page_key] = cp

    if show_timing_summary:
        render_timing_summary_metrics(dialogs, show_help=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button(
            "← Назад",
            key=f"{session_page_key}_prev",
            disabled=(cp == 0),
        ):
            st.session_state[session_page_key] = cp - 1
            st.rerun()
    with col2:
        start = cp * DIALOGS_PER_PAGE
        end = min(start + DIALOGS_PER_PAGE, n)
        st.markdown(
            f"**Страница {cp + 1} из {total_pages}** · {list_entity_caption} **{start + 1}–{end}** из **{n}** "
            f"(по **{DIALOGS_PER_PAGE}** на страницу)"
        )
    with col3:
        if st.button(
            "Вперёд →",
            key=f"{session_page_key}_next",
            disabled=(cp >= total_pages - 1),
        ):
            st.session_state[session_page_key] = cp + 1
            st.rerun()

    start_idx = cp * DIALOGS_PER_PAGE
    page = dialogs[start_idx : start_idx + DIALOGS_PER_PAGE]
    render_dialogs(
        page,
        disp_eval_mode,
        llm_eval_fields_disp,
        start_index=start_idx,
        expander_row_label=expander_row_label,
        expander_title_show_accuracy_icon=expander_title_show_accuracy_icon,
        show_dialog_level_api_response=show_dialog_level_api_response,
    )


def dialog_turn_count(record: dict) -> int:
    history = record.get("history") or []
    if isinstance(history, list):
        return len(history)
    return 0


def passes_turn_count_filter(n_turns: int, min_r: int, max_r: int) -> bool:
    """``min_r`` / ``max_r``: 0 — без ограничения с этой стороны."""
    if min_r > 0 and n_turns < min_r:
        return False
    if max_r > 0 and n_turns > max_r:
        return False
    return True


def filter_records_by_turn_count(
    records: List[dict],
    min_r: int,
    max_r: int,
) -> List[dict]:
    if min_r <= 0 and max_r <= 0:
        return list(records)
    return [
        r
        for r in records
        if passes_turn_count_filter(dialog_turn_count(r), min_r, max_r)
    ]


def _median_int(values: List[int]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def render_dialog_length_distribution_panel(
    records: List[dict],
    *,
    key_prefix: str,
    title: str = "📊 Длина диалогов",
    filter_caption: str = "Фильтр по числу реплик в history (0 — без ограничения с этой стороны).",
) -> Tuple[List[dict], int, int]:
    """
    Диаграмма распределения диалогов по числу реплик и фильтр min/max.
    Возвращает (отфильтрованные записи, min_turns, max_turns).
    """
    if title:
        st.markdown(f"#### {title}")
    if not records:
        st.info("Нет диалогов для статистики.")
        return [], 0, 0

    turn_counts = [dialog_turn_count(r) for r in records]
    total_turns = sum(turn_counts)
    tc_min = min(turn_counts)
    tc_max = max(turn_counts)
    tc_mean = total_turns / len(records)
    tc_med = _median_int(turn_counts)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Всего диалогов", len(records))
    with c2:
        st.metric("Всего реплик", total_turns)
    with c3:
        st.metric(
            "На диалог: мин / ср / медиана / макс",
            f"{tc_min} / {tc_mean:.1f} / {tc_med:.1f} / {tc_max}",
        )

    st.markdown(
        "**Распределение:** сколько **диалогов** имеют то или иное **число реплик** "
        "в поле `history`."
    )
    repl_hist = Counter(turn_counts)
    dist_rows = sorted(repl_hist.items(), key=lambda x: x[0])
    dist_df = pd.DataFrame(dist_rows, columns=["Число реплик", "Диалогов"])
    st.bar_chart(dist_df.set_index("Число реплик"), use_container_width=True)
    st.caption("Каждый столбец: столько диалогов содержит ровно указанное число реплик.")

    st.markdown("**Фильтр по длине**")
    st.caption(filter_caption)
    f1, f2 = st.columns(2)
    with f1:
        filter_min_turns = st.number_input(
            "Не меньше реплик",
            min_value=0,
            value=0,
            step=1,
            key=f"{key_prefix}_filter_min_turns",
            help="Минимум включительно. 0 — не отсекать снизу.",
        )
    with f2:
        filter_max_turns = st.number_input(
            "Не больше реплик",
            min_value=0,
            value=0,
            step=1,
            key=f"{key_prefix}_filter_max_turns",
            help="Максимум включительно. 0 — не отсекать сверху.",
        )

    min_i = int(filter_min_turns)
    max_i = int(filter_max_turns)
    filtered = filter_records_by_turn_count(records, min_i, max_i)
    if min_i > 0 or max_i > 0:
        st.info(
            f"Под фильтр попадает **{len(filtered)}** из **{len(records)}** диалогов "
            f"(по числу реплик в history)."
        )
    return filtered, min_i, max_i
