"""Общие хелперы для страниц бенчмарка."""

from __future__ import annotations

import os
import uuid
from typing import Dict

from benchmarking.runner import (
    BenchmarkConfig,
    RoleLLMConfig,
    normalize_parsed_case,
    parse_benchmark_cases_jsonl_text,
)


def roles_from_session(session_state: dict, supported_models: list[str]) -> Dict[str, RoleLLMConfig]:
    """Собирает конфигурации ролей из состояния Streamlit."""

    roles = ["assistant", "user", "evaluator"]
    return {
        r: RoleLLMConfig(
            model=session_state.get(f"{r}_model", supported_models[0]),
            api_key=session_state.get(f"{r}_api_key", ""),
            params_json=session_state.get(f"{r}_params_json", "{}"),
        )
        for r in roles
    }


def build_benchmark_config(
    *,
    session_state: dict,
    supported_models: list[str],
    mode: str,
    assistant_url: str,
    user_message_key: str,
    response_field_path: str,
    assistant_prompt: str,
    parse_json_response: bool,
    use_tools: bool,
    assistant_tools: str,
    user_prompt: str,
    max_turns: int,
    llm_delay: float,
    eval_mode: str,
    eval_field_path: str,
    custom_eval_code: str,
    evaluate_existing_only: bool,
    llm_eval_prompt: str,
    llm_eval_fields: str,
    exit_when_condition_met: bool,
    repeats_per_case: int = 1,
    repeats_stagger_sec: float = 1.0,
    semantic_pred_field_path: str = "theme",
    semantic_ref_field_path: str = "subtopic",
    semantic_similarity_threshold: float = 0.85,
    external_context_field_map_json: str = "",
    dm_base_url: str = "",
    dm_start_block_id: int = 0,
    dm_stop_at_block_id: int = 0,
    dm_first_user_phrase: str = "*начало диалога*",
    dm_add_data_json: str = "{}",
    dm_scenario_speaks_first: bool = False,
) -> BenchmarkConfig:
    """Собирает ``BenchmarkConfig`` из параметров формы."""

    return BenchmarkConfig(
        mode=mode,
        assistant_url=assistant_url,
        user_message_key=user_message_key,
        response_field_path=response_field_path,
        assistant_prompt=assistant_prompt,
        parse_json_response=parse_json_response,
        use_tools=use_tools,
        assistant_tools=assistant_tools,
        user_prompt=user_prompt,
        max_turns=max_turns,
        llm_delay=float(llm_delay),
        repeats_per_case=max(1, int(repeats_per_case)),
        repeats_stagger_sec=max(0.0, float(repeats_stagger_sec)),
        max_parallel_dialogs=1,
        eval_mode=eval_mode,
        eval_field_path=eval_field_path,
        custom_eval_code=custom_eval_code,
        evaluate_existing_only=evaluate_existing_only,
        llm_eval_prompt=llm_eval_prompt,
        llm_eval_fields=llm_eval_fields,
        semantic_pred_field_path=semantic_pred_field_path,
        semantic_ref_field_path=semantic_ref_field_path,
        semantic_similarity_threshold=float(semantic_similarity_threshold),
        exit_when_condition_met=exit_when_condition_met,
        external_context_field_map_json=external_context_field_map_json,
        external_session_id_field="id",
        external_coerce_int_fields_csv="",
        dm_base_url=dm_base_url.strip().rstrip("/"),
        dm_start_block_id=int(dm_start_block_id),
        dm_stop_at_block_id=int(dm_stop_at_block_id),
        dm_first_user_phrase=dm_first_user_phrase,
        dm_add_data_json=dm_add_data_json,
        dm_scenario_speaks_first=dm_scenario_speaks_first,
        litellm_api_base=(os.getenv("LITELLM_API_BASE") or "").strip(),
        roles=roles_from_session(session_state, supported_models),
    )


def bench_history_label(row: dict) -> str:
    """Форматирует строку истории запусков для выбора в UI."""

    title = (row.get("run_title") or "").strip()
    sid = str(row["id"])[:8]
    status = row["status"]
    when = row.get("created_at")
    when_text = str(when) if when is not None else ""
    if title:
        return f"📌 {title} | {status} | {when_text} | id {sid}…"
    return f"📌 (без названия) | {status} | {when_text} | id {sid}…"


def load_test_cases_from_jsonl_text(text: str) -> list:
    """Парсит JSONL с кейсами бенчмарка и нормализует их."""

    parsed, perr = parse_benchmark_cases_jsonl_text(text)
    if perr:
        raise ValueError(perr)
    cases = []
    for case in parsed:
        normalize_parsed_case(case)
        if "dialog_id" not in case:
            case["dialog_id"] = str(uuid.uuid4())
        cases.append(case)
    return cases
