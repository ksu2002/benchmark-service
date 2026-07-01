"""Бенчмарк агентов: загрузка трассировок Langfuse и оценка LLM-судьёй."""

from __future__ import annotations

import json
import os
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Бенчмарк агентов", layout="wide")

from benchmarking.runner import LLM_JUDGE_EVAL_MODE, summarize_context_fields
from ui.dialog_results_ui import render_benchmark_bootstrap_metrics, render_dialogs_paginated
from tools.export_langfuse_traces import fetch_langfuse_dialog_cases
from ui.judge_settings_ui import (
    DEFAULT_EVALUATOR_PARAMS,
    ensure_bench_judge_form_defaults,
    flush_pending_bench_judge,
    render_benchmark_judge_loader,
    render_benchmark_judge_save,
    render_judge_context_prompt_help,
    run_judge_test_on_case,
)
from common.security import redact_secrets_for_display
from integrations.litellm import get_model_names
from ui.sample_storage_ui import render_export_jsonl_actions

try:
    SUPPORTED_MODELS = get_model_names()
except Exception:
    SUPPORTED_MODELS = [os.getenv("LITELLM_MODEL_NAME", "openai/gpt-4o-mini")]

try:
    from storage.benchmark_backend import queue_backend_enabled, queue_backend_missing_vars

    _QUEUE_MISSING = queue_backend_missing_vars()
    QUEUE_AVAILABLE = queue_backend_enabled()
except ImportError:
    QUEUE_AVAILABLE = False
    _QUEUE_MISSING = ["установите зависимости: psycopg2-binary, pika, minio"]

_PAGE_KEY = "agentbench"
flush_pending_bench_judge(_PAGE_KEY)
ensure_bench_judge_form_defaults(_PAGE_KEY, use_tools=False)

if "evaluator_model" not in st.session_state:
    st.session_state["evaluator_model"] = SUPPORTED_MODELS[0]
if "evaluator_api_key" not in st.session_state:
    st.session_state["evaluator_api_key"] = os.getenv("LITELLM_API_KEY", "")
if "evaluator_params_json" not in st.session_state:
    st.session_state["evaluator_params_json"] = DEFAULT_EVALUATOR_PARAMS

if "agent_bench_cases" not in st.session_state:
    st.session_state["agent_bench_cases"] = []
if "agent_bench_results" not in st.session_state:
    st.session_state["agent_bench_results"] = []
if "agent_bench_fetch_info" not in st.session_state:
    st.session_state["agent_bench_fetch_info"] = {}


def _combine_date_time(d: date, t: time) -> datetime:
    return datetime.combine(d, t, tzinfo=timezone.utc)


def _dialog_key(case: dict) -> str:
    return str(case.get("dialog_id") or _thread_id(case) or "")


def _case_turns(case: dict) -> int:
    history = case.get("history") or []
    return sum(
        1
        for msg in history
        if isinstance(msg, dict) and msg.get("role") in ("user", "assistant", "tool")
    )


def _thread_id(case: dict) -> str:
    lf = case.get("langfuse") if isinstance(case.get("langfuse"), dict) else {}
    meta = lf.get("metadata") if isinstance(lf.get("metadata"), dict) else {}
    return str(meta.get("thread_id") or lf.get("session_id") or "")


def _build_judge_config() -> Dict[str, Any]:
    api_key = (st.session_state.get("evaluator_api_key") or "").strip()
    if not api_key:
        api_key = os.getenv("LITELLM_API_KEY", "")
    return {
        "evaluator": {
            "model": st.session_state.get("evaluator_model", SUPPORTED_MODELS[0]),
            "api_key": api_key,
            "params_json": st.session_state.get("evaluator_params_json", DEFAULT_EVALUATOR_PARAMS),
        },
        "llm_eval_prompt": st.session_state.get(f"{_PAGE_KEY}_llm_eval_prompt", ""),
        "llm_eval_fields": st.session_state.get(f"{_PAGE_KEY}_llm_eval_fields", "result,reason"),
        "assistant_prompt": "",
        "user_prompt": "",
        "assistant_tools": "[]",
        "evaluate_existing_only": True,
    }


def _merge_judge_into_case(case: dict, judge_out: dict) -> dict:
    row = dict(case)
    row.update(judge_out)
    row["accuracy"] = 1.0 if judge_out.get("result") else 0.0
    row["turns"] = _case_turns(case)
    row["mode"] = "Langfuse (агент)"
    row["eval_mode"] = LLM_JUDGE_EVAL_MODE
    return row


def _render_cases_summary(cases: List[dict], *, title: str) -> None:
    if not cases:
        st.info("Нет кейсов для отображения.")
        return
    st.markdown(f"##### {title}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Диалогов", len(cases))
    with c2:
        st.metric("Ср. реплик", f"{sum(_case_turns(c) for c in cases) / len(cases):.1f}")
    with c3:
        with_ctx = sum(
            1
            for c in cases
            if isinstance(c.get("context"), dict) and str(c["context"].get("eval") or "").strip()
        )
        st.metric("С context.eval", with_ctx)

    rows = []
    for case in cases:
        lf = case.get("langfuse") if isinstance(case.get("langfuse"), dict) else {}
        rows.append(
            {
                "dialog_id (= trace_id)": lf.get("trace_id") or case.get("dialog_id"),
                "conversation_key": (lf.get("conversation_key") or ""),
                "timestamp": lf.get("timestamp"),
                "name": lf.get("name"),
                "turns": _case_turns(case),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


st.title("🤖 Бенчмарк агентов")
st.caption(
    "Загрузка диалогов из Langfuse (полная history) и оценка **LLM-судьёй**."
)

tab_fetch, tab_cases, tab_judge, tab_results = st.tabs(
    ["1. Langfuse", "2. Кейсы", "3. LLM-судья", "4. Результаты"]
)

with tab_fetch:
    st.subheader("Подключение к Langfuse")
    c_host, c_pk, c_sk = st.columns([2, 1, 1])
    with c_host:
        lf_host = st.text_input(
            "Host",
            value=os.getenv("LANGFUSE_HOST", "https://[NDA_LANGFUSE_HOST]"),
            key="agent_lf_host",
        )
    with c_pk:
        lf_pk = st.text_input(
            "Public key",
            value=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            type="password",
            key="agent_lf_pk",
        )
    with c_sk:
        lf_sk = st.text_input(
            "Secret key",
            value=os.getenv("LANGFUSE_SECRET_KEY", ""),
            type="password",
            key="agent_lf_sk",
        )

    st.subheader("Период выгрузки")
    fd1, fd2 = st.columns(2)
    with fd1:
        date_from = st.date_input("С (UTC)", value=None, key="agent_lf_from")
        time_from = st.time_input(
            "Время с",
            value=time(0, 0),
            step=60,
            key="agent_lf_from_time",
        )
    with fd2:
        date_to = st.date_input("По (UTC)", value=None, key="agent_lf_to")
        time_to = st.time_input(
            "Время по",
            value=time(23, 59),
            step=60,
            key="agent_lf_to_time",
        )

    with st.expander("Дополнительные фильтры", expanded=False):
        tool_name_filter = st.text_input(
            "Инструмент (tool)",
            value="",
            key="agent_lf_tool",
            placeholder="check_building_ownership",
            help="Оставить пустым — все диалоги. Иначе только диалоги, где вызывался этот tool.",
        )

    fetch_btn = st.button("📥 Загрузить диалоги из Langfuse", type="primary", key="agent_fetch_btn")
    if fetch_btn:
        missing = [
            label
            for label, val in (
                ("Host", lf_host),
                ("Public key", lf_pk),
                ("Secret key", lf_sk),
            )
            if not (val or "").strip()
        ]
        if missing:
            st.error("Заполните: " + ", ".join(missing))
        elif not date_from and not date_to:
            st.error("Укажите хотя бы одну дату — «С» или «По».")
        else:
            from_ts = (
                _combine_date_time(date_from, time_from) if date_from else None
            )
            to_ts = (
                _combine_date_time(date_to, time_to) if date_to else None
            )

            try:
                with st.spinner("Загрузка диалогов из Langfuse…"):
                    cases = fetch_langfuse_dialog_cases(
                        host=lf_host.strip(),
                        public_key=lf_pk.strip(),
                        secret_key=lf_sk.strip(),
                        from_timestamp=from_ts,
                        to_timestamp=to_ts,
                        filter_tool_name=tool_name_filter.strip() or None,
                    )
                st.session_state["agent_bench_cases"] = cases
                st.session_state["agent_bench_results"] = []
                st.session_state["agent_bench_fetch_info"] = {
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "count": len(cases),
                    "filters": {
                        "from": str(from_ts) if from_ts else None,
                        "to": str(to_ts) if to_ts else None,
                        "tool": tool_name_filter.strip() or None,
                    },
                }
                st.success(f"Загружено диалогов: **{len(cases)}**")
            except Exception as e:
                st.error(f"Ошибка Langfuse: {e}")

    info = st.session_state.get("agent_bench_fetch_info") or {}
    if info:
        st.caption(
            f"Последняя загрузка: {info.get('fetched_at', '—')} · "
            f"кейсов: **{info.get('count', 0)}** · фильтры: `{info.get('filters', {})}`"
        )

with tab_cases:
    cases = st.session_state.get("agent_bench_cases") or []
    if not cases:
        st.info("Сначала загрузите трассировки на вкладке «Langfuse».")
    else:
        _render_cases_summary(cases, title="Сводка загруженных кейсов")
        ctx_summary = summarize_context_fields(cases)
        if ctx_summary:
            st.caption(
                "Поля context в выборке: "
                + ", ".join(f"`{k}`" for k in ctx_summary.keys())
            )
        cases_jsonl = "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n"
        render_export_jsonl_actions(
            cases_jsonl,
            key_prefix="agent_cases",
            case_count=len(cases),
            file_name="agent_bench_cases.jsonl",
            description="Кейсы бенчмарка агентов (Langfuse)",
            name_placeholder="agent-cases-v1",
        )
        st.markdown("---")
        st.subheader("Просмотр диалогов")
        render_dialogs_paginated(
            cases,
            LLM_JUDGE_EVAL_MODE,
            st.session_state.get(f"{_PAGE_KEY}_llm_eval_fields", "result,reason"),
            session_page_key="agent_cases_page_idx",
            expander_title_show_accuracy_icon=False,
            show_timing_summary=False,
        )

with tab_judge:
    cases = st.session_state.get("agent_bench_cases") or []
    st.subheader("Настройки LLM-судьи")
    render_benchmark_judge_loader(
        page_key=_PAGE_KEY,
        presets_available=QUEUE_AVAILABLE,
        queue_missing=_QUEUE_MISSING,
    )
    render_judge_context_prompt_help(cases)

    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        st.selectbox(
            "Модель оценщика",
            SUPPORTED_MODELS,
            key="evaluator_model",
        )
    with ec2:
        st.text_input(
            "API key оценщика",
            type="password",
            key="evaluator_api_key",
        )
    with ec3:
        st.text_input(
            "params_json",
            key="evaluator_params_json",
        )

    llm_eval_prompt = st.text_area(
        "Промпт LLM-судьи",
        height=200,
        key=f"{_PAGE_KEY}_llm_eval_prompt",
    )
    llm_eval_fields = st.text_input(
        "Поля JSON-ответа (через запятую)",
        help=(
            "Критерии оценки (0/1 или true/false): eval1,eval2,result — по каждому считается "
            "отдельная точность. Текстовые поля без влияния на accuracy: reason,details,comment."
        ),
        key=f"{_PAGE_KEY}_llm_eval_fields",
    )

    render_benchmark_judge_save(
        page_key=_PAGE_KEY,
        presets_available=QUEUE_AVAILABLE,
        queue_missing=_QUEUE_MISSING,
        evaluate_existing_only=True,
        default_api_key=os.getenv("LITELLM_API_KEY", ""),
    )

    if not cases:
        st.warning("Нет загруженных кейсов — сначала вкладка «Langfuse».")
    else:
        st.caption(f"Будет оценено кейсов: **{len(cases)}** (история из Langfuse, без генерации).")
        run_judge = st.button("🧑‍⚖️ Запустить LLM-судью", type="primary", key="agent_run_judge")
        if run_judge:
            if not llm_eval_prompt.strip():
                st.error("Заполните промпт LLM-судьи.")
            elif not (_build_judge_config().get("evaluator") or {}).get("api_key"):
                st.error("Укажите API key оценщика или задайте LITELLM_API_KEY в .env.")
            else:
                config = _build_judge_config()
                results: List[dict] = []
                progress = st.progress(0.0, text="Оценка…")
                err_count = 0
                first_err: Optional[str] = None
                for i, case in enumerate(cases):
                    try:
                        judge_out = run_judge_test_on_case(case, config)
                        results.append(_merge_judge_into_case(case, judge_out))
                    except Exception as e:
                        err_count += 1
                        if first_err is None:
                            first_err = str(e)
                        results.append(
                            _merge_judge_into_case(
                                case,
                                {"result": False, "reason": str(e), "raw_output": str(e)},
                            )
                        )
                    progress.progress(
                        (i + 1) / len(cases),
                        text=f"Оценка {i + 1} / {len(cases)}",
                    )
                progress.empty()
                st.session_state["agent_bench_results"] = results
                ok = sum(1 for r in results if r.get("accuracy") == 1.0)
                st.success(
                    f"Готово: **{ok}** / **{len(results)}** успешных"
                    + (f", ошибок вызова: **{err_count}**" if err_count else "")
                )
                if first_err:
                    st.error(f"Пример ошибки: {first_err}")

with tab_results:
    results = st.session_state.get("agent_bench_results") or []
    if not results:
        st.info("Результаты появятся после запуска LLM-судьи на вкладке «LLM-судья».")
    else:
        total = len(results)
        ok = sum(1 for r in results if r.get("accuracy") == 1.0)
        fail = total - ok
        st.subheader("Итоги оценки")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Точность", f"{ok / total:.1%}" if total else "—")
        with m2:
            st.metric("Успешных", ok)
        with m3:
            st.metric("Неуспешных", fail)

        llm_fields = st.session_state.get(f"{_PAGE_KEY}_llm_eval_fields", "result,reason")
        render_benchmark_bootstrap_metrics(
            results,
            llm_eval_fields=llm_fields,
            eval_mode=LLM_JUDGE_EVAL_MODE,
            key_prefix="agent_bootstrap",
        )

        results_jsonl = (
            "\n".join(
                json.dumps(redact_secrets_for_display(r), ensure_ascii=False, default=str)
                for r in results
            )
            + "\n"
        )
        render_export_jsonl_actions(
            results_jsonl,
            key_prefix="agent_results",
            case_count=len(results),
            file_name="agent_bench_results.jsonl",
            description="Результаты бенчмарка агентов",
            name_placeholder="agent-results-v1",
        )

        st.markdown("---")
        st.subheader("Все диалоги")
        render_dialogs_paginated(
            results,
            LLM_JUDGE_EVAL_MODE,
            llm_fields,
            session_page_key="agent_results_page_idx",
            show_timing_summary=False,
        )

        failures = [r for r in results if r.get("accuracy") == 0]
        if failures:
            st.subheader(f"Неуспешные ({len(failures)})")
            render_dialogs_paginated(
                failures,
                LLM_JUDGE_EVAL_MODE,
                llm_fields,
                session_page_key="agent_failures_page_idx",
                list_entity_caption="ошибки",
                expander_row_label="Диалог",
                expander_title_show_accuracy_icon=False,
                show_timing_summary=False,
            )
