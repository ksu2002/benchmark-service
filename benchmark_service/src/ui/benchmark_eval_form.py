"""Общие UI-блоки настроек оценки для страниц бенчмарка."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import streamlit as st

from benchmarking.runner import (
    LLM_JUDGE_EVAL_MODE,
    SEMANTIC_SIMILARITY_EVAL_MODE,
    is_llm_judge_eval_mode,
    is_semantic_similarity_eval_mode,
)
from ui.judge_settings_ui import ensure_bench_judge_form_defaults, render_benchmark_judge_loader


@dataclass
class BenchmarkEvalFormState:
    """Значения формы режима оценки."""

    custom_eval_code: str = ""
    evaluate_existing_only: bool = False
    llm_eval_prompt: str = ""
    llm_eval_fields: str = "result,reason"
    eval_field_path: str = ""
    semantic_pred_field_path: str = "response"
    semantic_ref_field_path: str = "subtopic"
    semantic_similarity_threshold: float = 0.85
    exit_when_condition_met: bool = False


def render_llm_judge_eval_block(
    *,
    page_key: str,
    eval_mode: str,
    queue_available: bool,
    queue_missing: Sequence[str],
    use_tools: bool,
    allow_existing_only: bool,
    existing_only_key: str,
    existing_only_help: str,
    existing_only_caption: str = "",
    default_prompt: str = "",
) -> tuple[str, str, bool]:
    """Рендерит блок LLM-судьи. Возвращает prompt, fields, evaluate_existing_only."""

    st.caption(
        "💡 Базовые плейсхолдеры: `{goals}`, `{history}`, `{assistant_prompt}`, "
        "`{available_tools}`, `{user_prompt}`. Плюс ключи из `context` кейса."
    )
    render_benchmark_judge_loader(
        page_key=page_key,
        presets_available=queue_available,
        queue_missing=list(queue_missing),
    )
    if default_prompt and f"{page_key}_llm_eval_prompt" not in st.session_state:
        st.session_state[f"{page_key}_llm_eval_prompt"] = default_prompt
    ensure_bench_judge_form_defaults(page_key, use_tools=use_tools)
    evaluate_existing_only = False
    if allow_existing_only:
        evaluate_existing_only = st.checkbox(
            "Оценивать существующие диалоги (без генерации диалога)",
            key=existing_only_key,
            help=existing_only_help,
        )
    elif existing_only_caption:
        st.caption(existing_only_caption)
    llm_eval_prompt = st.text_area(
        "Промпт LLM-судьи",
        height=180,
        key=f"{page_key}_llm_eval_prompt",
    )
    llm_eval_fields = st.text_input(
        "Ожидаемые поля в JSON-ответе (через запятую)",
        help=(
            "Критерии оценки (0/1 или true/false): eval1,eval2,result — по каждому считается "
            "отдельная точность. Текстовые поля без влияния на accuracy: reason,details,comment."
        ),
        key=f"{page_key}_llm_eval_fields",
    )
    return llm_eval_prompt, llm_eval_fields, evaluate_existing_only


def render_semantic_similarity_eval_block(
    *,
    pred_default: str,
    ref_default: str = "subtopic",
) -> tuple[str, str, float]:
    """Рендерит блок семантического сходства."""

    semantic_pred_field_path = st.text_input(
        "Путь к полю в последнем ответе ассистента",
        value=pred_default,
        help="Точка-нотация в full_response; для plain text — response или content.",
    )
    semantic_ref_field_path = st.text_input(
        "Путь к полю эталона в кейсе",
        value=ref_default,
    )
    semantic_similarity_threshold = st.slider(
        "Порог сходства (accuracy = 1, если cosine ≥ порога)",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.01,
    )
    return semantic_pred_field_path, semantic_ref_field_path, semantic_similarity_threshold


def render_custom_eval_block(*, default_code: str) -> str:
    """Рендерит поле кастомного Python-критерия."""

    st.caption(
        "💡 Доступные переменные: `goals`, `response`, `eval_value`, `history`, `context`. "
        "Разрешены вызовы len/any/all/str/bool/isinstance."
    )
    return st.text_area(
        "Введите выражение (True/False):",
        value=default_code,
        help="Доступны: goals, response, eval_value, history, context",
    )


def render_exit_when_condition_met(*, key: str = "bench_exit_when_condition_met") -> bool:
    """Чекбокс досрочного выхода при выполнении критерия."""

    return st.checkbox(
        "Выход, когда достигли условия",
        value=False,
        help="После ответа ассистента, при котором критерий уже выполняется, "
        "диалог завершается до лимита ходов.",
        key=key,
    )


def eval_mode_supports_early_exit(eval_mode: str) -> bool:
    """Возвращает ``True``, если для режима доступен ранний выход."""

    return eval_mode in (
        "Сравнить цель с полем из ответа",
        "Кастомный критерий (Python)",
        "Проверка вызова тулзов",
    )


__all__ = [
    "BenchmarkEvalFormState",
    "LLM_JUDGE_EVAL_MODE",
    "SEMANTIC_SIMILARITY_EVAL_MODE",
    "eval_mode_supports_early_exit",
    "is_llm_judge_eval_mode",
    "is_semantic_similarity_eval_mode",
    "render_custom_eval_block",
    "render_exit_when_condition_met",
    "render_llm_judge_eval_block",
    "render_semantic_similarity_eval_block",
]
