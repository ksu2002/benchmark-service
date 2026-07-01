"""Общие хелперы для страницы настроек LLM-судьи."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import streamlit as st

from benchmarking.runner import (
    LLMRuntimeContext,
    RoleLLMConfig,
    _goals_from_case,
    case_template_strings_for_eval,
    evaluate_dialog_with_llm,
    format_history_for_prompt,
    format_prompt_template,
    format_tools_for_prompt,
    merge_template_vars,
    parse_benchmark_cases_jsonl_text,
    summarize_context_fields,
)

DEFAULT_EVALUATOR_PARAMS = '{"temperature": 0.0, "max_tokens": 300}'

PRESET_KIND_JUDGE = "judge"
PRESET_KIND_PROMPT = "prompt"

PROMPT_WITH_TOOLS = """Вы — эксперт по качеству диалогов.

Цель пользователя: {goals}

Доступные ассистенту инструменты:
{available_tools}

История диалога:
{history}

Оцените, достиг ли ассистент цели пользователя.
Если цель требует вызова функции — проверьте:
- Была ли вызвана подходящая функция?
- Корректны ли переданные аргументы?
- Соответствует ли вызов описанию и параметрам функции?

Верните JSON с полями: result (true/false), reason (строка)."""

PROMPT_PLAIN = """Вы — эксперт по качеству диалогов.
Цель пользователя: {goals}

История диалога:
{history}

Оцените, достиг ли ассистент цели пользователя.
Верните JSON с полями: result (true/false), reason (строка)."""


def default_prompt(*, use_tools: bool) -> str:
    return PROMPT_WITH_TOOLS if use_tools else PROMPT_PLAIN


def init_judge_session_state(
    supported_models: List[str],
    *,
    default_api_key: str = "",
    default_model: str = "",
) -> None:
    model = default_model or (supported_models[0] if supported_models else "")
    if "judge_evaluator_model" not in st.session_state:
        st.session_state["judge_evaluator_model"] = model
    if "judge_evaluator_api_key" not in st.session_state:
        st.session_state["judge_evaluator_api_key"] = default_api_key
    if "judge_evaluator_params_json" not in st.session_state:
        st.session_state["judge_evaluator_params_json"] = DEFAULT_EVALUATOR_PARAMS
    if "judge_llm_eval_prompt" not in st.session_state:
        st.session_state["judge_llm_eval_prompt"] = default_prompt(use_tools=False)
    if "judge_llm_eval_fields" not in st.session_state:
        st.session_state["judge_llm_eval_fields"] = "result,reason"
    if "judge_evaluate_existing_only" not in st.session_state:
        st.session_state["judge_evaluate_existing_only"] = True
    if "judge_use_tools" not in st.session_state:
        st.session_state["judge_use_tools"] = False
    if "judge_assistant_prompt" not in st.session_state:
        st.session_state["judge_assistant_prompt"] = ""
    if "judge_user_prompt" not in st.session_state:
        st.session_state["judge_user_prompt"] = ""
    if "judge_assistant_tools" not in st.session_state:
        st.session_state["judge_assistant_tools"] = "[]"
    if "judge_test_cases" not in st.session_state:
        st.session_state["judge_test_cases"] = []
    if "judge_active_preset_name" not in st.session_state:
        st.session_state["judge_active_preset_name"] = ""
    if "judge_cal_pool" not in st.session_state:
        st.session_state["judge_cal_pool"] = []
    if "judge_cal_items" not in st.session_state:
        st.session_state["judge_cal_items"] = []
    if "judge_cal_sample_indices" not in st.session_state:
        st.session_state["judge_cal_sample_indices"] = []
    if "judge_cal_num_to_annotate" not in st.session_state:
        st.session_state["judge_cal_num_to_annotate"] = 0
    if "judge_cal_sample_seed" not in st.session_state:
        st.session_state["judge_cal_sample_seed"] = 42
    if "_judge_sample_meta_name" not in st.session_state:
        legacy = st.session_state.pop("judge_cal_sample_name", "")
        st.session_state["_judge_sample_meta_name"] = legacy if isinstance(legacy, str) else ""
    if "judge_active_sample_id" not in st.session_state:
        st.session_state["judge_active_sample_id"] = ""
    if "judge_cal_label_mode" not in st.session_state:
        st.session_state["judge_cal_label_mode"] = "binary"
    if "judge_cal_binary_criteria" not in st.session_state:
        st.session_state["judge_cal_binary_criteria"] = "goal_met,tool_ok,style_ok"
    if "judge_cal_ordinal_min" not in st.session_state:
        st.session_state["judge_cal_ordinal_min"] = 1
    if "judge_cal_ordinal_max" not in st.session_state:
        st.session_state["judge_cal_ordinal_max"] = 5
    if "judge_cal_categories" not in st.session_state:
        st.session_state["judge_cal_categories"] = "да,нет"
    if "judge_cal_llm_field" not in st.session_state:
        st.session_state["judge_cal_llm_field"] = "result"
    if "judge_cal_page" not in st.session_state:
        st.session_state["judge_cal_page"] = 0
    if "judge_cal_model_results" not in st.session_state:
        st.session_state["judge_cal_model_results"] = {}
    if "judge_cal_preset_results" not in st.session_state:
        st.session_state["judge_cal_preset_results"] = {}
    if "judge_cal_compare_results" not in st.session_state:
        st.session_state["judge_cal_compare_results"] = {}
    if "judge_cal_run_model" not in st.session_state:
        st.session_state["judge_cal_run_model"] = st.session_state.get(
            "judge_evaluator_model", ""
        )
    if "judge_cal_rank_strategy" not in st.session_state:
        st.session_state["judge_cal_rank_strategy"] = "worst_kappa"
    if "judge_cal_annot_split_view" not in st.session_state:
        st.session_state["judge_cal_annot_split_view"] = "all"
    if "judge_cal_split_test_ratio" not in st.session_state:
        st.session_state["judge_cal_split_test_ratio"] = 0.2
    if "judge_cal_split_seed" not in st.session_state:
        st.session_state["judge_cal_split_seed"] = 42


def preset_kind(config: Dict[str, Any]) -> str:
    kind = (config or {}).get("preset_kind")
    if kind == PRESET_KIND_PROMPT:
        return PRESET_KIND_PROMPT
    return PRESET_KIND_JUDGE


def is_prompt_preset_config(config: Dict[str, Any]) -> bool:
    return preset_kind(config) == PRESET_KIND_PROMPT


def current_judge_config_dict() -> Dict[str, Any]:
    cfg = {
        "preset_kind": PRESET_KIND_JUDGE,
        "evaluator": {
            "model": st.session_state.get("judge_evaluator_model", ""),
            "api_key": st.session_state.get("judge_evaluator_api_key", ""),
            "params_json": st.session_state.get(
                "judge_evaluator_params_json", DEFAULT_EVALUATOR_PARAMS
            ),
        },
        "llm_eval_prompt": st.session_state.get("judge_llm_eval_prompt", ""),
        "llm_eval_fields": st.session_state.get("judge_llm_eval_fields", "result,reason"),
        "evaluate_existing_only": bool(
            st.session_state.get("judge_evaluate_existing_only", True)
        ),
        "use_tools": bool(st.session_state.get("judge_use_tools", False)),
        "assistant_prompt": st.session_state.get("judge_assistant_prompt", ""),
        "user_prompt": st.session_state.get("judge_user_prompt", ""),
        "assistant_tools": st.session_state.get("judge_assistant_tools", "[]"),
    }
    return cfg


def current_prompt_config_dict() -> Dict[str, Any]:
    return {
        "preset_kind": PRESET_KIND_PROMPT,
        "llm_eval_prompt": st.session_state.get("judge_llm_eval_prompt", ""),
        "use_tools": bool(st.session_state.get("judge_use_tools", False)),
        "assistant_prompt": st.session_state.get("judge_assistant_prompt", ""),
        "user_prompt": st.session_state.get("judge_user_prompt", ""),
        "assistant_tools": st.session_state.get("judge_assistant_tools", "[]"),
    }


def schedule_apply_judge_config_dict(
    config: Dict[str, Any],
    *,
    preset_name: str = "",
) -> None:
    """Отложенное применение конфига — до отрисовки виджетов на следующем rerun."""
    st.session_state["_pending_judge_config"] = {
        "config": config,
        "preset_name": preset_name,
    }


def schedule_apply_judge_prompt_dict(
    config: Dict[str, Any],
    *,
    preset_name: str = "",
) -> None:
    st.session_state["_pending_judge_prompt_config"] = {
        "config": config,
        "preset_name": preset_name,
    }


def schedule_apply_judge_saved(
    config: Dict[str, Any],
    *,
    preset_name: str = "",
) -> None:
    if is_prompt_preset_config(config):
        schedule_apply_judge_prompt_dict(config, preset_name=preset_name)
    else:
        schedule_apply_judge_config_dict(config, preset_name=preset_name)


def schedule_judge_evaluator_model(model: str) -> None:
    """Отложенная смена модели — до отрисовки selectbox на следующем rerun."""
    st.session_state["_pending_judge_evaluator_model"] = model


def schedule_judge_cal_run_model(model: str) -> None:
    """Отложенная смена модели прогона на вкладке калибровки."""
    st.session_state["_pending_judge_cal_run_model"] = model


def flush_pending_judge_state() -> None:
    """Применить отложенные изменения настроек судьи (вызывать до виджетов)."""
    pending_prompt = st.session_state.pop("_pending_judge_prompt_config", None)
    if pending_prompt:
        apply_judge_prompt_dict(
            pending_prompt.get("config") or {},
            preset_name=pending_prompt.get("preset_name") or "",
        )
    pending_cfg = st.session_state.pop("_pending_judge_config", None)
    if pending_cfg:
        apply_judge_config_dict(
            pending_cfg.get("config") or {},
            preset_name=pending_cfg.get("preset_name") or "",
        )
    pending_model = st.session_state.pop("_pending_judge_evaluator_model", None)
    if pending_model:
        st.session_state["judge_evaluator_model"] = pending_model
    pending_cal_model = st.session_state.pop("_pending_judge_cal_run_model", None)
    if pending_cal_model:
        st.session_state["judge_cal_run_model"] = pending_cal_model


def apply_judge_config_dict(config: Dict[str, Any], *, preset_name: str = "") -> None:
    ev = config.get("evaluator") or {}
    st.session_state["judge_evaluator_model"] = ev.get("model") or st.session_state.get(
        "judge_evaluator_model", ""
    )
    st.session_state["judge_evaluator_api_key"] = ev.get("api_key") or ""
    st.session_state["judge_evaluator_params_json"] = ev.get(
        "params_json"
    ) or DEFAULT_EVALUATOR_PARAMS
    st.session_state["judge_llm_eval_prompt"] = config.get("llm_eval_prompt") or default_prompt(
        use_tools=bool(config.get("use_tools"))
    )
    st.session_state["judge_llm_eval_fields"] = (
        config.get("llm_eval_fields") or "result,reason"
    )
    st.session_state["judge_evaluate_existing_only"] = bool(
        config.get("evaluate_existing_only", True)
    )
    st.session_state["judge_use_tools"] = bool(config.get("use_tools", False))
    st.session_state["judge_assistant_prompt"] = config.get("assistant_prompt") or ""
    st.session_state["judge_user_prompt"] = config.get("user_prompt") or ""
    st.session_state["judge_assistant_tools"] = config.get("assistant_tools") or "[]"
    st.session_state["judge_active_preset_name"] = preset_name


def apply_judge_prompt_dict(config: Dict[str, Any], *, preset_name: str = "") -> None:
    st.session_state["judge_llm_eval_prompt"] = config.get("llm_eval_prompt") or default_prompt(
        use_tools=bool(config.get("use_tools"))
    )
    st.session_state["judge_use_tools"] = bool(config.get("use_tools", False))
    st.session_state["judge_assistant_prompt"] = config.get("assistant_prompt") or ""
    st.session_state["judge_user_prompt"] = config.get("user_prompt") or ""
    st.session_state["judge_assistant_tools"] = config.get("assistant_tools") or "[]"
    if preset_name:
        st.session_state["judge_active_preset_name"] = preset_name


def apply_judge_to_benchmark_session(config: Optional[Dict[str, Any]] = None) -> None:
    """Копирует настройки оценщика в ключи, которые читает страница бенчмарка."""
    cfg = config or current_judge_config_dict()
    ev = cfg.get("evaluator") or {}
    st.session_state["evaluator_model"] = ev.get("model") or st.session_state.get(
        "judge_evaluator_model", ""
    )
    st.session_state["evaluator_api_key"] = ev.get("api_key") or ""
    st.session_state["evaluator_params_json"] = ev.get("params_json") or DEFAULT_EVALUATOR_PARAMS
    st.session_state["judge_applied_llm_eval_prompt"] = cfg.get("llm_eval_prompt") or ""
    st.session_state["judge_applied_llm_eval_fields"] = (
        cfg.get("llm_eval_fields") or "result,reason"
    )
    st.session_state["judge_applied_evaluate_existing_only"] = bool(
        cfg.get("evaluate_existing_only", True)
    )
    st.session_state["judge_applied_preset_name"] = st.session_state.get(
        "judge_active_preset_name", ""
    )


def saved_judge_preset_label(preset: dict) -> str:
    cfg = preset.get("config") or {}
    kind = "Промпт" if is_prompt_preset_config(cfg) else "Судья"
    return f"{kind}: {preset.get('name') or '—'}"


def ensure_bench_judge_form_defaults(page_key: str, *, use_tools: bool = False) -> None:
    if f"{page_key}_llm_eval_prompt" not in st.session_state:
        st.session_state[f"{page_key}_llm_eval_prompt"] = default_prompt(
            use_tools=use_tools
        )
    if f"{page_key}_llm_eval_fields" not in st.session_state:
        st.session_state[f"{page_key}_llm_eval_fields"] = "result,reason"


def _bench_evaluate_existing_only_key(page_key: str) -> str:
    return {
        "abench": "bench_llm_eval_no_gen",
        "dmbench": "bench_dm_llm_eval_no_gen",
    }.get(page_key, f"{page_key}_llm_eval_no_gen")


def apply_judge_preset_to_benchmark_page(
    config: Dict[str, Any],
    *,
    preset_name: str = "",
    page_key: str,
) -> None:
    """Применить сохранённого судью/промпт к форме бенчмарка (session_state)."""
    if not is_prompt_preset_config(config):
        ev = config.get("evaluator") or {}
        if ev.get("model"):
            st.session_state["evaluator_model"] = ev["model"]
        if ev.get("api_key"):
            st.session_state["evaluator_api_key"] = ev["api_key"]
        if ev.get("params_json"):
            st.session_state["evaluator_params_json"] = ev["params_json"]
        st.session_state[_bench_evaluate_existing_only_key(page_key)] = bool(
            config.get("evaluate_existing_only", True)
        )
    st.session_state[f"{page_key}_llm_eval_prompt"] = config.get(
        "llm_eval_prompt"
    ) or default_prompt(use_tools=bool(config.get("use_tools")))
    st.session_state[f"{page_key}_llm_eval_fields"] = (
        config.get("llm_eval_fields") or "result,reason"
    )
    st.session_state[f"{page_key}_loaded_judge_name"] = preset_name


def schedule_apply_bench_judge_preset(
    config: Dict[str, Any],
    *,
    preset_name: str = "",
    page_key: str,
) -> None:
    st.session_state["_pending_bench_judge"] = {
        "config": config,
        "preset_name": preset_name,
        "page_key": page_key,
    }


def flush_pending_bench_judge(page_key: str) -> None:
    pending = st.session_state.get("_pending_bench_judge")
    if not pending or pending.get("page_key") != page_key:
        return
    st.session_state.pop("_pending_bench_judge", None)
    apply_judge_preset_to_benchmark_page(
        pending.get("config") or {},
        preset_name=pending.get("preset_name") or "",
        page_key=page_key,
    )


def bench_current_judge_config_dict(
    page_key: str,
    *,
    evaluate_existing_only: bool = True,
    use_tools: bool = False,
    default_api_key: str = "",
) -> Dict[str, Any]:
    """Конфиг судьи из session_state страницы бенчмарка (для сохранения пресета)."""
    api_key = (st.session_state.get("evaluator_api_key") or "").strip()
    if not api_key:
        api_key = (default_api_key or "").strip()
    ev_only_key = _bench_evaluate_existing_only_key(page_key)
    if ev_only_key in st.session_state:
        evaluate_existing_only = bool(
            st.session_state.get(ev_only_key, evaluate_existing_only)
        )
    return {
        "preset_kind": PRESET_KIND_JUDGE,
        "evaluator": {
            "model": st.session_state.get("evaluator_model", ""),
            "api_key": api_key,
            "params_json": st.session_state.get(
                "evaluator_params_json", DEFAULT_EVALUATOR_PARAMS
            ),
        },
        "llm_eval_prompt": st.session_state.get(f"{page_key}_llm_eval_prompt", ""),
        "llm_eval_fields": st.session_state.get(
            f"{page_key}_llm_eval_fields", "result,reason"
        ),
        "evaluate_existing_only": evaluate_existing_only,
        "use_tools": use_tools,
        "assistant_prompt": "",
        "user_prompt": "",
        "assistant_tools": "[]",
    }


def bench_current_prompt_config_dict(
    page_key: str,
    *,
    use_tools: bool = False,
) -> Dict[str, Any]:
    """Только промпт судьи из session_state страницы бенчмарка."""
    return {
        "preset_kind": PRESET_KIND_PROMPT,
        "llm_eval_prompt": st.session_state.get(f"{page_key}_llm_eval_prompt", ""),
        "use_tools": use_tools,
        "assistant_prompt": "",
        "user_prompt": "",
        "assistant_tools": "[]",
    }


def render_benchmark_judge_loader(
    *,
    page_key: str,
    presets_available: bool,
    queue_missing: Optional[List[str]] = None,
) -> None:
    loaded = (st.session_state.get(f"{page_key}_loaded_judge_name") or "").strip()
    if loaded:
        st.caption(f"Загружено: **{loaded}**")

    if not presets_available:
        st.info(
            "Загрузка судьи из БД недоступна — настройте очередь или задайте промпт вручную. "
            f"Не хватает: **{', '.join(queue_missing or [])}**"
        )
        return

    from storage.benchmark_backend import get_judge_preset, list_judge_presets

    presets = list_judge_presets()
    if not presets:
        st.caption(
            "Нет сохранённых пресетов — сохраните конфигурацию ниже или на странице «Настройки LLM-судьи»."
        )
        return

    preset_by_id = {str(p["id"]): p for p in presets}
    preset_ids = list(preset_by_id.keys())
    c1, c2 = st.columns([3, 1])
    with c1:
        selected = st.selectbox(
            "Сохранённая конфигурация",
            options=[""] + preset_ids,
            format_func=lambda pid: (
                "— не выбрано —"
                if not pid
                else saved_judge_preset_label(preset_by_id[pid])
            ),
            key=f"{page_key}_judge_preset_select",
        )
    with c2:
        if st.button(
            "Загрузить",
            disabled=not selected,
            key=f"{page_key}_judge_load_btn",
        ):
            loaded_row = get_judge_preset(selected)
            if loaded_row:
                schedule_apply_bench_judge_preset(
                    loaded_row.get("config") or {},
                    preset_name=loaded_row.get("name") or "",
                    page_key=page_key,
                )
                st.rerun()


def render_benchmark_judge_save(
    *,
    page_key: str,
    presets_available: bool,
    queue_missing: Optional[List[str]] = None,
    evaluate_existing_only: bool = True,
    use_tools: bool = False,
    default_api_key: str = "",
) -> None:
    """Форма сохранения судьи или промпта в Postgres (как на странице настроек)."""
    st.markdown("##### Сохранить конфигурацию")
    st.caption(
        "Сохраните **судью** (модель, API key, промпт, поля ответа) или только **промпт**. "
        "Пресеты доступны на всех страницах бенчмарка."
    )
    if not presets_available:
        st.info(
            "Сохранение в БД недоступно — настройте очередь. "
            f"Не хватает: **{', '.join(queue_missing or [])}**"
        )
        return

    from storage.benchmark_backend import save_judge_preset

    with st.form(f"{page_key}_judge_save_form"):
        save_kind = st.radio(
            "Что сохранить",
            [PRESET_KIND_JUDGE, PRESET_KIND_PROMPT],
            format_func=lambda k: (
                "Судья (модель + промпт + поля)"
                if k == PRESET_KIND_JUDGE
                else "Промпт"
            ),
            horizontal=True,
        )
        save_name = st.text_input("Имя")
        save_desc = st.text_input("Описание")
        if st.form_submit_button("💾 Сохранить", type="primary"):
            if not save_name.strip():
                st.error("Укажите имя")
            else:
                cfg = (
                    bench_current_prompt_config_dict(page_key, use_tools=use_tools)
                    if save_kind == PRESET_KIND_PROMPT
                    else bench_current_judge_config_dict(
                        page_key,
                        evaluate_existing_only=evaluate_existing_only,
                        use_tools=use_tools,
                        default_api_key=default_api_key,
                    )
                )
                save_judge_preset(save_name.strip(), cfg, description=save_desc)
                st.session_state[f"{page_key}_loaded_judge_name"] = save_name.strip()
                st.success(f"Сохранено: **{save_name.strip()}**")
                st.rerun()


def build_llm_context_from_judge_config(config: Dict[str, Any]) -> LLMRuntimeContext:
    ev = config.get("evaluator") or {}
    roles = {
        "evaluator": RoleLLMConfig(
            model=ev.get("model") or "",
            api_key=ev.get("api_key") or "",
            params_json=ev.get("params_json") or DEFAULT_EVALUATOR_PARAMS,
        )
    }
    return LLMRuntimeContext(
        roles=roles,
        api_base=os.getenv("LITELLM_API_BASE"),
        default_model=os.getenv("LITELLM_MODEL_NAME", "openai/gpt-4o-mini"),
    )


def parse_tools_json(raw: str) -> List[Dict]:
    s = (raw or "").strip() or "[]"
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def build_prompt_preview(
    case: dict,
    *,
    prompt_template: str,
    assistant_prompt: str = "",
    user_prompt: str = "",
    assistant_tools: str = "[]",
) -> str:
    tools = parse_tools_json(assistant_tools)
    tools_desc = format_tools_for_prompt(tools)
    history = case.get("history") or []
    vars_map = merge_template_vars(
        case_template_strings_for_eval(case),
        goals=str(case.get("goals") or ""),
        history=format_history_for_prompt(history),
        assistant_prompt=assistant_prompt,
        user_prompt=user_prompt,
        available_tools=tools_desc,
    )
    return format_prompt_template(prompt_template, vars_map)


def run_judge_test_on_case(
    case: dict,
    config: Dict[str, Any],
) -> dict:
    prompt = config.get("llm_eval_prompt") or ""
    fields_raw = config.get("llm_eval_fields") or "result,reason"
    expected_fields = [f.strip() for f in fields_raw.split(",") if f.strip()]
    if not expected_fields:
        expected_fields = ["result", "reason"]
    tools = parse_tools_json(config.get("assistant_tools") or "[]")
    llm_ctx = build_llm_context_from_judge_config(config)
    return evaluate_dialog_with_llm(
        goals=_goals_from_case(case),
        history=case.get("history") or [],
        eval_prompt_template=prompt,
        expected_fields=expected_fields,
        llm_ctx=llm_ctx,
        assistant_prompt=config.get("assistant_prompt") or "",
        user_prompt=config.get("user_prompt") or "",
        available_tools_description=format_tools_for_prompt(tools),
        extra_template_vars=case_template_strings_for_eval(case),
    )


def render_judge_context_prompt_help(cases: List[dict]) -> None:
    """Подсказка по плейсхолдерам context — как на странице бенчмарка."""
    st.caption(
        "В промпте доступны `{goals}`, `{history}` и ключи из поля `context` кейса "
        "(например `{schedule}`). У диалогов без поля подставляется пустая строка."
    )
    if not cases:
        st.caption(
            "Загрузите датасет на вкладке «Датасет», чтобы увидеть список полей `context` "
            "в вашей выборке."
        )
        return

    ctx_summary = summarize_context_fields(cases)
    with_context = sum(
        1 for c in cases if isinstance(c.get("context"), dict) and c["context"]
    )
    if ctx_summary:
        keys_line = ", ".join(f"`{{{k}}}`" for k in ctx_summary)
        st.info(
            f"В датасете **{with_context}** из **{len(cases)}** диалогов с непустым `context`. "
            f"Поля для подстановки в промпт: {keys_line}. "
            "У диалогов без поля в `context` подставляется пустая строка."
        )
        with st.expander("Context в промпте — примеры значений (до 3 на поле)", expanded=False):
            for field, examples in ctx_summary.items():
                st.markdown(f"**`{field}`**")
                for i, ex in enumerate(examples, 1):
                    st.caption(f"Пример {i}")
                    st.code(ex)
    else:
        st.caption(
            "В загруженном датасете нет непустого `context`. В JSONL можно задать объект "
            "`context` (строки, числа, вложенные объекты) и использовать `{имя_поля}` в промпте."
        )


def load_cases_from_text(text: str) -> tuple[List[dict], Optional[str]]:
    return parse_benchmark_cases_jsonl_text(text)
