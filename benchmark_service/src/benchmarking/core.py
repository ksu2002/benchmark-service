"""
Чистая логика бенчмарка (без Streamlit). Используется страницей и воркером.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import threading
import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import litellm
import requests

from benchmarking.evaluation.custom_eval import (
    build_custom_eval_context,
    evaluate_custom_eval_code,
)
from common.utils import (
    ASSISTANT_ROLES_REPLY_TIMING,
    USER_ROLES_REPLY_TIMING,
    benchmark_mean_reply_times,
    enrich_benchmark_results_timing_inplace,
    role_for_reply_timing,
    row_avg_reply_times_from_history,
    turn_reply_duration_sec,
)

from benchmarking.config_models import (
    BENCHMARK_HTTP_CONNECT_TIMEOUT_SEC,
    BENCHMARK_HTTP_READ_TIMEOUT_SEC,
    BENCHMARK_HTTP_TIMEOUT,
    BenchmarkConfig,
    LLM_JUDGE_EVAL_MODE,
    LLMRuntimeContext,
    RoleLLMConfig,
    SEMANTIC_SIMILARITY_EVAL_MODE,
    benchmark_config_from_dict,
    benchmark_config_to_dict,
    benchmark_criterion_accuracy_summary,
    benchmark_mean_criterion_score,
    coerce_llm_criterion_passed,
    derive_criterion_scores,
    is_llm_judge_eval_mode,
    is_semantic_similarity_eval_mode,
    llm_eval_scoring_field_names,
    parse_llm_eval_fields,
    row_criterion_accuracy,
    row_mean_criterion_score,
    _force_llm_judge_fail,
)
from benchmarking.parsing.cases import (
    case_context_as_str_dict,
    case_context_raw,
    case_template_strings_for_eval,
    cases_to_assistant_gen_form_rows,
    cases_to_dm_bench_gen_form_rows,
    common_context_from_cases,
    dm_stop_block_id_from_case,
    effective_case_context_dict,
    format_prompt_template,
    goals_text_from_case,
    history_from_case,
    merge_template_vars,
    normalize_parsed_case,
    parse_benchmark_cases_json_text,
    parse_benchmark_cases_jsonl_text,
    resolve_case_prompt_templates,
    summarize_context_fields,
)


log = logging.getLogger(__name__)


def _wall_iso() -> str:
    """Метка UTC для логов таймлайна (человекочитаемо)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_external_context_field_map_json(raw: str) -> Optional[Dict[str, str]]:
    """
    None — класть в POST весь объект context (как без маппинга).
    dict (в т.ч. пустой {}) — только перечисленные пары: имя поля в JSON тела → ключ в context.
    """
    s = (raw or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        log.warning(
            "external_context_field_map_json: невалидный JSON, в POST уходит полный context"
        )
        return None
    if not isinstance(obj, dict):
        return None
    out: Dict[str, str] = {}
    for k, v in obj.items():
        if isinstance(k, str) and isinstance(v, str) and v.strip():
            out[k] = v.strip()
    return out


def build_external_extra_body(
    case: dict, mapping: Optional[Dict[str, str]]
) -> Dict[str, Any]:
    ctx = case_context_raw(case)
    if mapping is None:
        return dict(ctx)
    out: Dict[str, Any] = {}
    for api_key, ctx_key in mapping.items():
        if ctx_key in ctx:
            out[api_key] = ctx[ctx_key]
    return out


def _get_nested_value(data: Any, path: str):
    if not path.strip():
        return data
    keys = path.strip().split(".")
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and key.isdigit():
            idx = int(key)
            if 0 <= idx < len(current):
                current = current[idx]
            else:
                return "[Поле не найдено]"
        else:
            return "[Поле не найдено]"
    return current


def _field_path_to_text(data: object, path: str) -> str:
    if not (path or "").strip():
        return ""
    val = _get_nested_value(data, path)
    if val == "[Поле не найдено]" or val is None:
        return ""
    return str(val).strip()


def _last_assistant_turn(history: list) -> Optional[dict]:
    for msg in reversed(history or []):
        if msg.get("role") == "assistant":
            return msg
    return None


def _resolve_semantic_pred_text(cfg: BenchmarkConfig, result: dict) -> str:
    """Текст pred только из последней реплики assistant в history."""
    path = (cfg.semantic_pred_field_path or cfg.eval_field_path or "response").strip()
    last = _last_assistant_turn(result.get("history"))
    if not last:
        return ""
    full_resp = last.get("full_response") or {}
    if isinstance(full_resp, dict):
        text = _field_path_to_text(full_resp, path)
        if text:
            return text
    content = last.get("content")
    if path in ("response", "content", "") and content:
        return str(content).strip()
    return ""


def _resolve_semantic_ref_text(cfg: BenchmarkConfig, case: dict) -> str:
    path = (cfg.semantic_ref_field_path or "subtopic").strip()
    return _field_path_to_text(case, path)


def _run_llm_eval_for_case(
    *,
    cfg: BenchmarkConfig,
    case: dict,
    goals: str,
    result: dict,
    llm_ctx: LLMRuntimeContext,
    available_tools_description: str,
    case_key: str,
) -> dict:
    expected_fields = [f.strip() for f in cfg.llm_eval_fields.split(",") if f.strip()]
    hist_list = _normalize_history_for_eval(result.get("history"))
    if not hist_list:
        hist_list = history_from_case(case)
    history_text = format_history_for_prompt(hist_list)
    _, _, eval_vars = resolve_case_prompt_templates(
        case,
        cfg,
        goals,
        history_text=history_text,
        available_tools_description=available_tools_description,
    )
    ap_for_eval = (
        eval_vars["assistant_prompt"]
        if cfg.mode == "LLM (через LiteLLM)"
        else (
            "[Сценарий Dialog Manager]"
            if cfg.mode == "Сценарий (Dialog Manager)"
            else "[Внешний ассистент]"
        )
    )
    t_eval0 = time.monotonic()
    log.info(
        "BENCH timeline wall=%s phase=llm_eval_begin dialog_id=%s "
        "history_turns=%s (часто основная пауза между диалогами — ожидание оценщика)",
        _wall_iso(),
        case_key,
        len(hist_list),
    )
    llm_eval_result = evaluate_dialog_with_llm(
        goals=goals,
        history=hist_list,
        eval_prompt_template=cfg.llm_eval_prompt,
        expected_fields=expected_fields,
        llm_ctx=llm_ctx,
        assistant_prompt=ap_for_eval,
        user_prompt=eval_vars["user_prompt"],
        available_tools_description=available_tools_description,
        llm_delay=cfg.llm_delay,
        extra_template_vars=case_template_strings_for_eval(case),
    )
    log.info(
        "BENCH timeline wall=%s phase=llm_eval_done dialog_id=%s llm_eval_monotonic_sec=%.3f",
        _wall_iso(),
        case_key,
        time.monotonic() - t_eval0,
    )
    return llm_eval_result


def parse_external_coerce_int_field_names(raw: str) -> List[str]:
    """Список имён полей из строки «a, b» или «a;b»."""
    s = (raw or "").strip()
    if not s:
        return []
    normalized = s.replace(";", ",")
    return [x.strip() for x in normalized.split(",") if x.strip()]


def _coerce_external_api_numeric_fields(
    body: Dict[str, Any], field_names: Sequence[str]
) -> None:
    """Схемы вроде flat_id: int — при строке из context часто 422 у FastAPI."""
    for name in field_names:
        v = body.get(name)
        if v is None or isinstance(v, bool) or isinstance(v, int):
            continue
        if isinstance(v, str) and v.strip():
            try:
                body[name] = int(v.strip(), 10)
            except ValueError:
                pass
        elif isinstance(v, float) and v == int(v):
            body[name] = int(v)


class ExternalAssistant:
    def __init__(
        self,
        url: str,
        session_id: str,
        response_field_path: str = "response",
        user_message_key: str = "content",
        session_id_field: str = "id",
        coerce_int_field_names: Optional[Tuple[str, ...]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        self.url = url
        self.id = session_id
        self.response_field_path = response_field_path
        self.user_message_key = (user_message_key or "message").strip() or "message"
        self.session_id_field = (session_id_field or "id").strip() or "id"
        self.coerce_int_field_names = coerce_int_field_names or ()
        self.extra_body = extra_body or {}

    def reply(self, history: List[Dict[str, str]]) -> Dict[str, Any]:
        last_user_msg = None
        for msg in reversed(history):
            if msg.get("role") == "user":
                last_user_msg = msg
                break
        user_text = last_user_msg.get("content", "") if last_user_msg else ""
        # Сначала поля из context (маппинг), затем перезаписываем id диалога и реплику пользователя.
        request_body: Dict[str, Any] = dict(self.extra_body)
        request_body[self.session_id_field] = self.id
        request_body[self.user_message_key] = user_text
        _coerce_external_api_numeric_fields(request_body, self.coerce_int_field_names)
        try:
            resp = requests.post(self.url, json=request_body, timeout=BENCHMARK_HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            displayed = _get_nested_value(data, self.response_field_path)
            return {
                "response": str(displayed),
                "eval_value": str(displayed),
                "full_response": data,
            }
        except requests.exceptions.HTTPError as e:
            detail = ""
            if e.response is not None:
                try:
                    detail = (e.response.text or "").strip()[:800]
                except Exception:
                    pass
            suffix = f" — ответ сервера: {detail}" if detail else ""
            error_msg = f"[Ошибка внешнего ассистента: {e}{suffix}]"
            return {"response": error_msg, "eval_value": error_msg, "full_response": {}}
        except Exception as e:
            error_msg = f"[Ошибка внешнего ассистента: {str(e)}]"
            return {"response": error_msg, "eval_value": error_msg, "full_response": {}}


class LLMAssistant:
    def __init__(
        self,
        system_prompt: str,
        parse_json: bool = False,
        tools: Optional[List[Dict]] = None,
        llm_ctx: Optional[LLMRuntimeContext] = None,
    ):
        self.system_prompt = system_prompt
        self.parse_json = parse_json
        self.tools = tools or []
        self.llm_ctx = llm_ctx

    def reply(self, history: List[Dict[str, str]]) -> Dict[str, Any]:
        api_history = copy.deepcopy(history)
        messages = [{"role": "system", "content": self.system_prompt}] + api_history
        try:
            if not self.llm_ctx:
                raise RuntimeError("LLMRuntimeContext не задан")
            kwargs = self.llm_ctx.litellm_kwargs("assistant")
            if self.tools:
                kwargs["tools"] = self.tools
                kwargs["tool_choice"] = "auto"

            response = litellm.completion(messages=messages, **kwargs)
            message = response.choices[0].message

            full_response = {"response": message.content or ""}
            if hasattr(message, "tool_calls") and message.tool_calls:
                full_response["tool_calls"] = [
                    tc.to_dict() for tc in message.tool_calls
                ]

            eval_value = message.content or ""
            if self.parse_json and eval_value:
                try:
                    parsed = json.loads(eval_value)
                    full_response.update(parsed)
                except json.JSONDecodeError:
                    pass

            return {
                "response": eval_value,
                "eval_value": eval_value,
                "full_response": full_response,
            }
        except Exception as e:
            error_msg = f"[Ошибка LLM: {str(e)}]"
            return {"response": error_msg, "eval_value": error_msg, "full_response": {}}


def _litellm_assistant_text(message: Any) -> str:
    """Текст из ответа LiteLLM: строка или список content-блоков (новый API)."""
    raw = getattr(message, "content", None)
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: List[str] = []
        for p in raw:
            if isinstance(p, dict) and p.get("text") is not None:
                parts.append(str(p["text"]))
        if parts:
            return "\n".join(parts)
    return str(raw)


def swap_roles(messages):
    role_map = {"user": "assistant", "assistant": "user"}
    out: List[Dict[str, str]] = []
    for msg in messages:
        role = role_map.get(msg.get("role"), msg.get("role", "user"))
        c = msg.get("content")
        if c is None:
            c = ""
        elif not isinstance(c, str):
            c = str(c)
        out.append({"role": role, "content": c})
    return out


class LLMUserSimulatorMultilabel:
    def __init__(self, system_prompt: str, llm_ctx: LLMRuntimeContext):
        self.system_prompt = system_prompt
        self.llm_ctx = llm_ctx

    def reply(self, history: List[Dict[str, str]]) -> str:
        swapped = swap_roles(history)
        messages = [{"role": "system", "content": self.system_prompt}] + swapped
        try:
            response = litellm.completion(
                messages=messages, **self.llm_ctx.litellm_kwargs("user")
            )
            message = response.choices[0].message
            text = _litellm_assistant_text(message).strip()
            if not text and getattr(message, "tool_calls", None):
                text = "Хорошо."
            if not text:
                text = "Слушаю."
            # Вся реплика уходит в form_next_phrase: раньше брали только split("\n")[0],
            # из‑за чего после «Привет!» с переводом строки в DM попадала одна первая строка.
            return text
        except Exception as e:
            return f"[Ошибка симулятора: {str(e)}]"


def _is_assistant_role_for_turn(role: Any) -> bool:
    if role is None:
        return False
    r = str(role).strip().lower()
    return r in ASSISTANT_ROLES_REPLY_TIMING


def _backfill_missing_reply_durations_in_history(history: Any, sim_sec: float) -> None:
    """
    Если на репликах нет reply_duration_sec, но известен wall симуляции диалога,
    распределяем оставшееся время поровну по «дырявым» ходам (до LLM-оценки).
    """
    if not isinstance(history, list) or sim_sec < 0:
        return
    idx_roles: List[int] = []
    for i, m in enumerate(history):
        if not isinstance(m, dict):
            continue
        rr = role_for_reply_timing(m)
        if rr in USER_ROLES_REPLY_TIMING or rr in ASSISTANT_ROLES_REPLY_TIMING:
            idx_roles.append(i)
    if not idx_roles:
        return
    missing = [
        i for i in idx_roles if turn_reply_duration_sec(history[i]) is None
    ]
    if not missing:
        return
    existing_sum = sum(
        (turn_reply_duration_sec(history[i]) or 0.0) for i in idx_roles
    )
    budget = max(0.0, float(sim_sec) - float(existing_sum))
    share = (budget / len(missing)) if budget > 0 else 0.0
    for i in missing:
        history[i]["reply_duration_sec"] = round(share, 4)


class DialogSessionMultilabel:
    def __init__(
        self,
        user_simulator,
        assistant,
        max_turns: int = 5,
        is_external: bool = False,
        llm_delay: float = 0.0,
    ):
        self.user = user_simulator
        self.assistant = assistant
        self.history = []
        self.turns = 0
        self.llm_delay = llm_delay
        self.max_turns = max_turns
        self.is_external = is_external

    def run_from_existing_history(
        self,
        initial_history: List[Dict],
        should_stop_after_assistant: Optional[Callable[[List[Dict]], bool]] = None,
        *,
        assistant_speaks_first: bool = False,
    ):
        self.history = [dict(m) for m in initial_history]
        bot_resp: Dict[str, Any] = {
            "response": "",
            "eval_value": None,
            "full_response": {},
        }
        if len(self.history) <= 0:
            if assistant_speaks_first:
                if self.llm_delay > 0:
                    time.sleep(self.llm_delay)
                t0 = time.monotonic()
                bot_resp = self.assistant.reply(self.history)
                as_dt = time.monotonic() - t0
                self.history.append(
                    {
                        "role": "assistant",
                        "content": bot_resp["response"],
                        "full_response": bot_resp.get("full_response", {}),
                        "eval_value": bot_resp.get("eval_value"),
                        "is_external": self.is_external,
                        "reply_duration_sec": as_dt,
                    }
                )
                if should_stop_after_assistant and should_stop_after_assistant(
                    self.history
                ):
                    self.turns = 1
                    return {
                        "predicted_intents": bot_resp["response"],
                        "eval_value": bot_resp["eval_value"],
                        "full_response": bot_resp["full_response"],
                        "turns": self.turns,
                        "history": self.history,
                    }
            else:
                t0 = time.monotonic()
                user_msg = self.user.reply(self.history)
                user_dt = time.monotonic() - t0
                self.history.append(
                    {
                        "role": "user",
                        "content": user_msg,
                        "reply_duration_sec": user_dt,
                    }
                )

        while self.turns < self.max_turns:
            self.turns += 1
            last_role = self.history[-1].get("role") if self.history else None
            if _is_assistant_role_for_turn(last_role) or last_role is None:
                if self.llm_delay > 0:
                    time.sleep(self.llm_delay)
                t0 = time.monotonic()
                user_msg = self.user.reply(self.history)
                user_dt = time.monotonic() - t0
                self.history.append(
                    {
                        "role": "user",
                        "content": user_msg,
                        "reply_duration_sec": user_dt,
                    }
                )
            else:
                if self.llm_delay > 0:
                    time.sleep(self.llm_delay)
                t0 = time.monotonic()
                bot_resp = self.assistant.reply(self.history)
                as_dt = time.monotonic() - t0
                self.history.append(
                    {
                        "role": "assistant",
                        "content": bot_resp["response"],
                        "full_response": bot_resp.get("full_response", {}),
                        "eval_value": bot_resp.get("eval_value"),
                        "is_external": self.is_external,
                        "reply_duration_sec": as_dt,
                    }
                )
                if should_stop_after_assistant and should_stop_after_assistant(
                    self.history
                ):
                    break

        return {
            "predicted_intents": bot_resp["response"],
            "eval_value": bot_resp["eval_value"],
            "full_response": bot_resp["full_response"],
            "turns": self.turns,
            "history": self.history,
        }

    def run_semantic_eval_session(
        self,
        initial_history: List[Dict],
        *,
        assistant_speaks_first: bool = False,
    ):
        """
        Режим «Семантическое сходство»: ровно одна новая реплика assistant для оценки.

        - Пустая history: симулятор пользователя → один ответ ассистента → стоп.
        - History из кейса: подставляется как контекст; если последняя реплика user —
          один ответ ассистента; если assistant — ещё одна реплика user (симулятор), затем assistant.
        """
        self.history = [dict(m) for m in initial_history]
        self.turns = 0
        bot_resp: Dict[str, Any] = {
            "response": "",
            "eval_value": None,
            "full_response": {},
        }

        if not self.history:
            if assistant_speaks_first:
                if self.llm_delay > 0:
                    time.sleep(self.llm_delay)
                t0 = time.monotonic()
                bot_resp = self.assistant.reply(self.history)
                as_dt = time.monotonic() - t0
                self.history.append(
                    {
                        "role": "assistant",
                        "content": bot_resp["response"],
                        "full_response": bot_resp.get("full_response", {}),
                        "eval_value": bot_resp.get("eval_value"),
                        "is_external": self.is_external,
                        "reply_duration_sec": as_dt,
                    }
                )
                self.turns = 1
                return {
                    "predicted_intents": bot_resp["response"],
                    "eval_value": bot_resp["eval_value"],
                    "full_response": bot_resp["full_response"],
                    "turns": self.turns,
                    "history": self.history,
                }
            t0 = time.monotonic()
            user_msg = self.user.reply(self.history)
            user_dt = time.monotonic() - t0
            self.history.append(
                {
                    "role": "user",
                    "content": user_msg,
                    "reply_duration_sec": user_dt,
                }
            )
        elif _is_assistant_role_for_turn(self.history[-1].get("role")):
            t0 = time.monotonic()
            user_msg = self.user.reply(self.history)
            user_dt = time.monotonic() - t0
            self.history.append(
                {
                    "role": "user",
                    "content": user_msg,
                    "reply_duration_sec": user_dt,
                }
            )

        if self.llm_delay > 0:
            time.sleep(self.llm_delay)
        t0 = time.monotonic()
        bot_resp = self.assistant.reply(self.history)
        as_dt = time.monotonic() - t0
        self.history.append(
            {
                "role": "assistant",
                "content": bot_resp["response"],
                "full_response": bot_resp.get("full_response", {}),
                "eval_value": bot_resp.get("eval_value"),
                "is_external": self.is_external,
                "reply_duration_sec": as_dt,
            }
        )
        self.turns = 1
        return {
            "predicted_intents": bot_resp["response"],
            "eval_value": bot_resp["eval_value"],
            "full_response": bot_resp["full_response"],
            "turns": self.turns,
            "history": self.history,
        }


def _turn_text_for_prompt(turn: dict) -> str:
    c = turn.get("content")
    if c is not None and str(c).strip() != "":
        return str(c)
    for k in ("text", "message", "body"):
        v = turn.get(k)
        if v is not None and str(v).strip() != "":
            return str(v)
    fr = turn.get("full_response")
    if isinstance(fr, dict):
        r = fr.get("response")
        if r is not None and str(r).strip() != "":
            return str(r)
    return ""


def format_history_for_prompt(history: Optional[List[Dict]]) -> str:
    if not history:
        return ""
    lines: List[str] = []
    for turn in history:
        if not isinstance(turn, dict):
            continue
        role_raw = turn.get("role")
        if role_raw is None:
            continue
        role = str(role_raw).strip().lower()
        content = _turn_text_for_prompt(turn) or ""

        if role in ("user", "human", "client"):
            lines.append(f"Пользователь: {content}")
        elif role in ("assistant", "operator", "bot", "assistant_bot", "model"):
            full_resp = turn.get("full_response", {}) or {}
            if not isinstance(full_resp, dict):
                full_resp = {}
            tool_calls = full_resp.get("tool_calls") or []
            tool_lines: List[str] = []
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    func = tc.get("function", {}) or {}
                    name = func.get("name", "unknown")
                    args_str = func.get("arguments", "{}")
                    try:
                        args_dict = json.loads(args_str)
                        args_pretty = json.dumps(
                            args_dict, ensure_ascii=False, indent=2
                        )
                    except Exception:
                        args_pretty = str(args_str)
                    tool_lines.append(
                        f"Ассистент → вызвал функцию: {name} с аргументами:\n{args_pretty}"
                    )
            if content.strip():
                lines.append(f"Ассистент: {content}")
            lines.extend(tool_lines)
            if not content.strip() and not tool_lines:
                if full_resp.get("scenario_finished"):
                    lines.append(
                        "Ассистент: [сценарий завершён без озвучиваемой реплики]"
                    )
                else:
                    lines.append(
                        "Ассистент: [ответ без озвучиваемого текста]"
                    )
        elif role == "tool":
            name = turn.get("name", "tool")
            lines.append(f"[Результат {name}]: {content}")
        else:
            lines.append(f"{role_raw}: {content}")
    return "\n".join(lines)


def extract_json_object_from_llm_text(text: str) -> Optional[dict]:
    """Извлекает JSON-объект из ответа LLM (в т.ч. из блока ```json ... ```)."""
    raw = (text or "").strip()
    if not raw:
        return None
    candidates: List[str] = []
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        candidates.append(brace.group(0).strip())
    seen: set[str] = set()
    for chunk in candidates:
        if chunk in seen:
            continue
        seen.add(chunk)
        try:
            parsed = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def derive_eval_result(parsed: dict, expected_fields: List[str]) -> bool:
    """Итог оценки: все scoring-поля из llm_eval_fields должны быть пройдены."""
    scoring = llm_eval_scoring_field_names(expected_fields)
    if not scoring:
        return False
    scores = derive_criterion_scores(parsed, scoring)
    if not scores:
        checks: List[bool] = []
        for field in expected_fields:
            val = parsed.get(field)
            if isinstance(val, dict) and "passed" in val:
                checks.append(bool(val.get("passed")))
        if checks:
            return all(checks)
        if "result" in parsed:
            return bool(parsed.get("result"))
        return False
    for field in scoring:
        if field not in scores:
            scores[field] = False
    return all(scores.values())


def format_eval_field_for_display(value: Any) -> str:
    """Человекочитаемое значение поля LLM-оценки (в т.ч. вложенный rubric)."""
    if value is None:
        return "—"
    if isinstance(value, dict):
        if "passed" in value or "score" in value or "failed_checks" in value:
            status = "✅" if value.get("passed") else "❌"
            parts = [status]
            if value.get("score") is not None:
                parts.append(f"score={value['score']}")
            failed = value.get("failed_checks") or []
            if failed:
                parts.append(
                    "failed: " + "; ".join(str(x) for x in failed if str(x).strip())
                )
            return " · ".join(parts) if len(parts) > 1 else parts[0]
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except TypeError:
            return str(value)
    if isinstance(value, (list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    text = str(value).strip()
    return text if text else "—"


def build_eval_result_from_parsed(
    parsed: dict,
    expected_fields: List[str],
    *,
    raw_output: str,
) -> dict:
    """Собирает результат оценки из распарсенного JSON."""
    out: dict = {field: parsed[field] for field in expected_fields if field in parsed}
    out["raw_output"] = raw_output
    scoring = llm_eval_scoring_field_names(expected_fields)
    scores = derive_criterion_scores(parsed, scoring)
    for field in scoring:
        if field not in scores:
            scores[field] = False
    out["criterion_scores"] = scores
    out["criterion_accuracy"] = {
        field: (1.0 if passed else 0.0) for field, passed in scores.items()
    }
    out["result"] = derive_eval_result(parsed, expected_fields)
    if "reason" in parsed and "reason" not in out:
        out["reason"] = parsed["reason"]
    return out


def evaluate_dialog_with_llm(
    goals: str,
    history: List[Dict],
    eval_prompt_template: str,
    expected_fields: List[str],
    llm_ctx: LLMRuntimeContext,
    assistant_prompt: str = "",
    user_prompt: str = "",
    available_tools_description: str = "Нет доступных инструментов.",
    llm_delay: float = 0.0,
    extra_template_vars: Optional[dict] = None,
) -> dict:
    """
    Вызов LiteLLM для оценщика.
    """
    history = _normalize_history_for_eval(history)
    history_text = format_history_for_prompt(history)
    vars_map = merge_template_vars(
        extra_template_vars or {},
        goals=goals,
        history=history_text,
        assistant_prompt=assistant_prompt,
        user_prompt=user_prompt,
        available_tools=available_tools_description,
    )
    prompt = format_prompt_template(eval_prompt_template, vars_map)

    last_raw = ""
    for attempt in range(3):
        if attempt == 0 and llm_delay > 0:
            time.sleep(llm_delay)
        try:
            _eval_kw = llm_ctx.litellm_kwargs("evaluator")
            if attempt == 0:
                log.info(
                    "LLM evaluator: model=%s api_base=%s prompt_len=%s",
                    _eval_kw.get("model"),
                    _eval_kw.get("api_base") or "(env default / OpenAI)",
                    len(prompt),
                )
            response = litellm.completion(
                messages=[{"role": "user", "content": prompt}],
                **_eval_kw,
            )
            raw_output = _litellm_assistant_text(
                response.choices[0].message
            ).strip()
            last_raw = raw_output

            parsed = extract_json_object_from_llm_text(raw_output)
            if isinstance(parsed, dict):
                missing = [f for f in expected_fields if f not in parsed]
                if not missing:
                    return build_eval_result_from_parsed(
                        parsed, expected_fields, raw_output=raw_output
                    )
            if attempt < 2:
                continue

            if isinstance(parsed, dict):
                if parsed:
                    return build_eval_result_from_parsed(
                        parsed, expected_fields, raw_output=raw_output
                    )
            result = "true" in raw_output.lower() or "вердикт: да" in raw_output.lower()
            return {"result": result, "raw_output": raw_output}
        except Exception as e:
            if attempt < 2:
                last_raw = str(e)
                continue
            return {"result": False, "raw_output": str(e)}
    return {"result": False, "raw_output": last_raw}


def format_tools_for_prompt(tools: List[Dict]) -> str:
    if not tools:
        return "Нет доступных инструментов."

    parts = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "Без описания")
        params = func.get("parameters", {})

        try:
            params_str = json.dumps(params, ensure_ascii=False, indent=2)
        except Exception:
            params_str = str(params)

        parts.append(
            f"● {name}\n" f"  Описание: {desc}\n" f"  Параметры:\n{params_str}"
        )

    return "\n\n".join(parts)


def _goals_from_case(case: dict) -> str:
    raw_goals = case.get("goals", [])
    if isinstance(raw_goals, str):
        s = raw_goals.strip()
        goals_list = [s] if s else []
    elif isinstance(raw_goals, list):
        goals_list = []
        for g in raw_goals:
            if g is None:
                continue
            s = str(g).strip()
            if s:
                goals_list.append(s)
    else:
        goals_list = []
    return ", ".join(goals_list) if goals_list else "Без цели"


def _tool_eval_condition_met(case: dict, history: List[Dict]) -> bool:
    """Та же логика, что финальная оценка в режиме «Проверка вызова тулзов»."""
    goal_set = set(case.get("goals", []))
    tool_calls: List = []
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            full_resp = msg.get("full_response", {})
            if isinstance(full_resp, dict) and "tool_calls" in full_resp:
                tool_calls = full_resp["tool_calls"]
                break
    if goal_set == {"None"}:
        return len(tool_calls) == 0
    called_names = {
        tc.get("function", {}).get("name")
        for tc in tool_calls
        if isinstance(tc, dict)
    }
    return bool(goal_set & called_names)


def _compare_or_custom_eval_met(
    cfg: BenchmarkConfig, case: dict, goals: str, history: List[Dict]
) -> bool:
    """Та же логика, что ветка custom_eval_code после диалога (сравнение / кастом)."""
    last_assistant = None
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            last_assistant = msg
            break
    if not last_assistant:
        return False
    full_resp = last_assistant.get("full_response") or {}
    if (
        cfg.mode in ("Внешний URL", "Сценарий (Dialog Manager)")
        and cfg.eval_mode == "Сравнить цель с полем из ответа"
    ):
        eval_value = _get_nested_value(full_resp, cfg.eval_field_path)
    else:
        eval_value = last_assistant.get("eval_value")
        if eval_value is None:
            eval_value = last_assistant.get("content", "")
    eval_context = build_custom_eval_context(
        goals=goals,
        response=full_resp,
        eval_value=eval_value,
        history=history,
        context=case_context_raw(case),
    )
    try:
        return evaluate_custom_eval_code(cfg.custom_eval_code, eval_context)
    except Exception:
        return False


def _dm_user_phrases_for_semantic_eval(
    case: dict,
    cfg: BenchmarkConfig,
    user_sim: LLMUserSimulatorMultilabel,
    first_phrase: str,
) -> List[str]:
    """Список user-фраз для DM: replay history + при необходимости новая реплика симулятора."""
    case_hist = history_from_case(case)
    if not case_hist:
        if cfg.dm_scenario_speaks_first:
            return [first_phrase]
        return [first_phrase]

    phrases: List[str] = []
    for msg in case_hist:
        role = msg.get("role")
        if role_for_reply_timing(msg) in USER_ROLES_REPLY_TIMING or (
            isinstance(role, str) and role.strip().lower() == "user"
        ):
            content = msg.get("content")
            if content is not None and str(content).strip():
                phrases.append(str(content).strip())

    if not phrases:
        phrases = [first_phrase]
    elif _is_assistant_role_for_turn(case_hist[-1].get("role")):
        if cfg.llm_delay > 0:
            time.sleep(cfg.llm_delay)
        phrases.append(user_sim.reply(case_hist))
    return phrases


def _should_stop_after_assistant_fn(
    cfg: BenchmarkConfig, case: dict, goals: str
) -> Optional[Callable[[List[Dict]], bool]]:
    if not cfg.exit_when_condition_met:
        return None

    def check(history: List[Dict]) -> bool:
        # Принудительная остановка по dm_stop_at_block_id обрабатывается только в run_dm_scenario_session.
        if cfg.eval_mode == "Проверка вызова тулзов":
            return _tool_eval_condition_met(case, history)
        if cfg.eval_mode in (
            "Сравнить цель с полем из ответа",
            "Кастомный критерий (Python)",
        ):
            return _compare_or_custom_eval_met(cfg, case, goals, history)
        if cfg.eval_mode == "Достигнут блок (block_id)":
            return _block_goal_met(
                case,
                goals,
                history,
                dm_stop_default=int(cfg.dm_stop_at_block_id or 0),
            )
        return False

    return check


# ---------------------------------------------------------------------------
# NDA: реализация интеграции с Dialog Manager (HTTP API, эндпоинты, payload,
# разбор стриминга form_next_phrase, block_data/scenario_steps и т.д.) удалена.
# ---------------------------------------------------------------------------

_NDA_DM_MSG = (
    "NDA: интеграция с Dialog Manager недоступна в публичной версии репозитория"
)


def _dm_user_phrases_for_semantic_eval(
    case: dict,
    cfg: BenchmarkConfig,
    user_sim: LLMUserSimulatorMultilabel,
    first_phrase: str,
) -> List[str]:
    # NDA: логика replay history для DM-сценария скрыта
    raise NotImplementedError(_NDA_DM_MSG)


def _dm_operator_parse(resp: dict) -> Tuple[List[str], str]:
    # NDA: разбор поля results[] ответа Dialog Manager скрыт
    return [], ""


def _resolve_dm_start_block_id(case: dict, default: int) -> int:
    # NDA: правила резолва start_block_id из кейса скрыты
    return int(default or 0)


def _coerce_positive_block_id(val: Any) -> Optional[int]:
    try:
        n = int(val)
        return n if n > 0 else None
    except (TypeError, ValueError):
        return None


def _dm_block_id_from_response(resp: Any) -> Optional[int]:
    # NDA: извлечение block_data.id из ответа DM скрыто
    return None


def _dm_collect_block_ids_recursive(obj: Any, out: set[int], depth: int = 0) -> None:
    # NDA: рекурсивный обход структуры ответа DM скрыт
    return None


def _dm_block_ids_from_steps_blob(steps: Any) -> set[int]:
    # NDA
    return set()


def _dm_extra_block_ids_from_embedded_steps(resp: dict) -> set[int]:
    # NDA
    return set()


def _dm_predecessor_block_ids_for_root(resp: dict) -> set[int]:
    # NDA
    return set()


def _dm_all_block_ids_from_response(resp: Any) -> set[int]:
    # NDA
    return set()


def _dm_full_response_dict_roots(fr: dict) -> List[dict]:
    # NDA
    return [fr] if isinstance(fr, dict) else []


def _collect_block_ids_from_dm_history(history: List[Dict]) -> set[int]:
    # NDA
    return set()


def _parse_block_id_tokens(text: str) -> set[int]:
    out: set[int] = set()
    for m in re.finditer(r"\b(\d{1,9})\b", text or ""):
        try:
            out.add(int(m.group(1)))
        except ValueError:
            pass
    return out


def _goal_target_block_ids_explicit_only(case: dict, goals_str: str) -> set[int]:
    # NDA: правила целевых block_id из goals/context скрыты
    return set()


def _resolve_goal_block_id_for_metric(case: dict, form_default: Any) -> int:
    return int(form_default or 0)


def _parse_goal_block_ids(
    case: dict,
    goals_str: str,
    *,
    dm_stop_default: int = 0,
) -> set[int]:
    # NDA
    return set()


def _block_goal_met(
    case: dict,
    goals_str: str,
    history: List[Dict],
    *,
    dm_stop_default: int = 0,
) -> bool:
    # NDA: проверка достижения block_id в истории DM скрыта
    return False


def _dm_forced_stop_params(
    case: dict, cfg: BenchmarkConfig, goals_str: str
) -> Tuple[int, str]:
    # NDA
    return 0, ""


def _dm_stop_after_block_met(
    case: dict,
    cfg: BenchmarkConfig,
    history: List[Dict],
    goals_str: str,
) -> bool:
    # NDA
    return False


def _minimal_benchmark_result_from_history(history: List[Dict]) -> dict:
    last_op = ""
    for msg in reversed(history or []):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            last_op = str(msg.get("content") or "")
            break
    return {
        "predicted_intents": last_op,
        "eval_value": last_op,
        "full_response": {},
        "turns": len(history or []),
        "history": list(history or []),
    }


def _dm_substitute_in_data_structure(value: Any, variables: dict) -> Any:
    """Подстановка {var} в add_data — generic, без специфики DM API."""
    if isinstance(value, str):
        out = value
        for k, v in (variables or {}).items():
            out = out.replace("{" + str(k) + "}", str(v))
        return out
    if isinstance(value, dict):
        return {k: _dm_substitute_in_data_structure(v, variables) for k, v in value.items()}
    if isinstance(value, list):
        return [_dm_substitute_in_data_structure(x, variables) for x in value]
    return value


def _resolve_dm_stop_at_block_id(
    case: dict,
    form_default: Any,
    goals_str: str,
) -> int:
    # NDA
    try:
        return int(form_default or 0)
    except (TypeError, ValueError):
        return 0


def _dm_merge_add_data(case: dict, cfg: BenchmarkConfig) -> Optional[dict]:
    # NDA: сбор и merge add_data для create_dialog скрыт
    return None


def _dm_create_dialog(
    base_url: str,
    start_block_id: int,
    add_data: Optional[dict],
    session: Optional[requests.Session] = None,
    *,
    test_dialog: bool = True,
) -> str:
    # NDA: HTTP create_dialog — скрыто
    raise NotImplementedError(_NDA_DM_MSG)


def _dm_form_next_phrase(
    base_url: str,
    dialog_id: str,
    text: str,
    session: Optional[requests.Session] = None,
    *,
    turn_label: str = "",
) -> dict:
    # NDA: HTTP form_next_phrase (stream) — скрыто
    raise NotImplementedError(_NDA_DM_MSG)


def _dm_end_dialog_once(
    ended_guard: Dict[str, bool],
    base_url: str,
    dialog_id: str,
    result: str,
    result_type: str,
    error_data: Any = None,
    session: Optional[requests.Session] = None,
) -> Optional[dict]:
    # NDA: HTTP end_dialog — скрыто
    raise NotImplementedError(_NDA_DM_MSG)


def run_dm_scenario_session(
    case: dict,
    cfg: BenchmarkConfig,
    user_sim: LLMUserSimulatorMultilabel,
    goals: str,
    user_prompt_rendered: str,
    available_tools_description: str,
    stop_after_assistant: Optional[Callable[[List[Dict]], bool]] = None,
) -> dict:
    # NDA: полный цикл прогона сценария в Dialog Manager скрыт
    return {
        "predicted_intents": "",
        "eval_value": "",
        "full_response": {},
        "turns": 0,
        "history": [],
        "dm_session_error": _NDA_DM_MSG,
        "dm_session_error_type": "NotImplementedError",
    }


def merge_case_with_benchmark_output(
    case: dict,
    *,
    goals_text: str,
    result: dict,
    cfg: BenchmarkConfig,
    accuracy: float,
    extra_fields: dict,
) -> dict:
    """
    В results.jsonl попадают все поля исходного кейса (context, goals, …)
    плюс результаты прогона. При непустом ``cfg.scenario_id`` в строку записывается поле ``scenario_id``
    запуска (перекрывает одноимённое поле из кейса). Поля прогона перекрывают остальные одноимённые ключи из кейса
    (history, full_response, turns, …). goals_text — нормализованная строка целей для промптов.
    """
    out = dict(case)
    out["goals_text"] = goals_text
    out.update(
        {
            "predicted_intents": result["predicted_intents"],
            "eval_field_value": result.get("eval_value", ""),
            "accuracy": accuracy,
            "turns": result["turns"],
            "mode": cfg.mode,
            "history": result["history"],
            "full_response": result["full_response"],
        }
    )
    out.update(extra_fields)
    out["goals_text"] = goals_text
    out["predicted_intents"] = result["predicted_intents"]
    out["eval_field_value"] = result.get("eval_value", "")
    out["accuracy"] = accuracy
    out["turns"] = result["turns"]
    out["mode"] = cfg.mode
    out["history"] = result["history"]
    out["full_response"] = result["full_response"]
    dm_did = result.get("dm_dialog_id")
    if dm_did is not None and str(dm_did).strip():
        out["dialog_id"] = str(dm_did).strip()
    _sid = (getattr(cfg, "scenario_id", None) or "").strip()
    if _sid:
        out["scenario_id"] = _sid
    return out


def process_single_case(
    case: dict,
    cfg: BenchmarkConfig,
    llm_ctx: LLMRuntimeContext,
) -> dict:
    """Один диалог → запись результата (как раньше в session_state)."""
    goals = _goals_from_case(case)
    t_dialog_start: Optional[float] = None
    t_case0 = time.monotonic()
    case_key = str(case.get("dialog_id") or case.get("id") or case.get("name") or "?")
    log.info(
        "BENCH timeline wall=%s phase=case_begin dialog_id=%s mode=%r eval_mode=%r "
        "llm_delay_sec=%s max_turns=%s",
        _wall_iso(),
        case_key,
        cfg.mode,
        cfg.eval_mode,
        cfg.llm_delay,
        cfg.max_turns,
    )

    if is_llm_judge_eval_mode(cfg.eval_mode) and cfg.evaluate_existing_only:
        initial_history = history_from_case(case)
        result = {
            "predicted_intents": "",
            "eval_value": "",
            "full_response": {},
            "turns": len(
                [m for m in initial_history if m.get("role") == "assistant"]
            ),
            "history": initial_history,
        }
        available_tool_names: List = []
        available_tools_description = "Нет доступных инструментов."
        if (
            cfg.mode == "LLM (через LiteLLM)"
            and cfg.use_tools
            and cfg.assistant_tools.strip()
        ):
            try:
                tools_list = json.loads(cfg.assistant_tools)
                available_tool_names = [
                    tool.get("function", {}).get("name")
                    for tool in tools_list
                    if isinstance(tool, dict) and "function" in tool
                ]
                available_tools_description = format_tools_for_prompt(tools_list)
            except Exception:
                available_tool_names = []

    elif (
        cfg.eval_mode == "Достигнут блок (block_id)"
        and cfg.evaluate_existing_only
        and cfg.mode == "Сценарий (Dialog Manager)"
    ):
        initial_history = history_from_case(case)
        result = _minimal_benchmark_result_from_history(initial_history)

    else:
        t_dialog_start = time.monotonic()
        session_id = str(uuid.uuid4())
        tools = []
        available_tool_names = []
        available_tools_description = "Нет доступных инструментов."
        if cfg.mode == "LLM (через LiteLLM)" and cfg.use_tools and cfg.assistant_tools.strip():
            try:
                tools = json.loads(cfg.assistant_tools)
                available_tool_names = [
                    tool.get("function", {}).get("name")
                    for tool in tools
                    if isinstance(tool, dict) and "function" in tool
                ]
                available_tools_description = format_tools_for_prompt(tools)
            except json.JSONDecodeError as e:
                raise ValueError(f"Ошибка парсинга JSON инструментов: {e}") from e

        is_external = cfg.mode == "Внешний URL"
        ap_rendered, up_rendered, _ = resolve_case_prompt_templates(
            case,
            cfg,
            goals,
            history_text="",
            available_tools_description=available_tools_description,
        )
        if cfg.mode == "Сценарий (Dialog Manager)":
            user_sim = LLMUserSimulatorMultilabel(up_rendered, llm_ctx)
            stop_fn = _should_stop_after_assistant_fn(cfg, case, goals)
            result = run_dm_scenario_session(
                case,
                cfg,
                user_sim,
                goals,
                up_rendered,
                available_tools_description,
                stop_fn,
            )
        elif cfg.mode == "Внешний URL":
            field_map = parse_external_context_field_map_json(
                cfg.external_context_field_map_json
            )
            extra = build_external_extra_body(case, field_map)
            if cfg.external_unique_session_id:
                dialog_id = session_id
            else:
                dialog_id = str(case.get("dialog_id") or "").strip() or session_id
            _coerce_names = parse_external_coerce_int_field_names(
                cfg.external_coerce_int_fields_csv
            )
            assistant = ExternalAssistant(
                url=cfg.assistant_url,
                session_id=dialog_id,
                response_field_path=cfg.response_field_path.strip() or "response",
                user_message_key=cfg.user_message_key.strip() or "message",
                session_id_field=cfg.external_session_id_field.strip() or "id",
                coerce_int_field_names=tuple(_coerce_names),
                extra_body=extra,
            )
            user_sim = LLMUserSimulatorMultilabel(up_rendered, llm_ctx)
            session = DialogSessionMultilabel(
                user_sim,
                assistant=assistant,
                max_turns=cfg.max_turns,
                llm_delay=cfg.llm_delay,
                is_external=is_external,
            )
            initial_history = history_from_case(case)
            if is_semantic_similarity_eval_mode(cfg.eval_mode):
                result = session.run_semantic_eval_session(
                    initial_history,
                    assistant_speaks_first=cfg.assistant_speaks_first,
                )
            else:
                stop_fn = _should_stop_after_assistant_fn(cfg, case, goals)
                result = session.run_from_existing_history(
                    initial_history,
                    should_stop_after_assistant=stop_fn,
                    assistant_speaks_first=cfg.assistant_speaks_first,
                )
        else:
            assistant = LLMAssistant(
                ap_rendered,
                parse_json=cfg.parse_json_response,
                tools=tools,
                llm_ctx=llm_ctx,
            )
            user_sim = LLMUserSimulatorMultilabel(up_rendered, llm_ctx)
            session = DialogSessionMultilabel(
                user_sim,
                assistant=assistant,
                max_turns=cfg.max_turns,
                llm_delay=cfg.llm_delay,
                is_external=False,
            )
            initial_history = history_from_case(case)
            if is_semantic_similarity_eval_mode(cfg.eval_mode):
                result = session.run_semantic_eval_session(
                    initial_history,
                    assistant_speaks_first=cfg.assistant_speaks_first,
                )
            else:
                stop_fn = _should_stop_after_assistant_fn(cfg, case, goals)
                result = session.run_from_existing_history(
                    initial_history,
                    should_stop_after_assistant=stop_fn,
                    assistant_speaks_first=cfg.assistant_speaks_first,
                )

    t_dialog_end = time.monotonic()
    if t_dialog_start is not None:
        log.info(
            "BENCH timeline wall=%s phase=dialog_sim_done dialog_id=%s "
            "dialog_monotonic_sec=%.3f turns=%s dm_dialog_id=%r",
            _wall_iso(),
            case_key,
            t_dialog_end - t_dialog_start,
            result.get("turns"),
            result.get("dm_dialog_id"),
        )
        _backfill_missing_reply_durations_in_history(
            result.get("history"),
            float(t_dialog_end - t_dialog_start),
        )
    else:
        log.info(
            "BENCH timeline wall=%s phase=skip_live_dialog dialog_id=%s "
            "(evaluate_existing_only или ветка без симуляции)",
            _wall_iso(),
            case_key,
        )

    accuracy = 0.0
    extra_fields: dict = {}

    _dm_err = result.pop("dm_session_error", None)
    _dm_err_meta: Dict[str, Any] = {}
    if _dm_err is not None:
        _dm_err_meta["dm_session_error"] = _dm_err
        for _k in (
            "dm_session_error_type",
            "dm_session_error_http_status",
            "dm_session_error_response_body",
            "dm_scenario_api_timeout",
            "dm_scenario_api_timeout_sec",
            "dm_empty_operator_reply",
        ):
            _v = result.pop(_k, None)
            if _v is not None:
                _dm_err_meta[_k] = _v

    _dm_max_turns_exceeded = bool(result.pop("dm_max_turns_exceeded", False))
    _dm_fnp_calls = result.pop("dm_form_next_phrase_calls", None)

    if _dm_err is not None:
        extra_fields.update(_dm_err_meta)
        accuracy = 0.0

    # LLM-оценщик вызываем всегда в этом режиме, даже при ошибке сессии DM/ассистента,
    # чтобы в результате были raw_output / reason (иначе в UI пусто).
    if is_llm_judge_eval_mode(cfg.eval_mode):
        try:
            atd = available_tools_description
        except NameError:
            atd = "Нет доступных инструментов."
        llm_eval_result = _run_llm_eval_for_case(
            cfg=cfg,
            case=case,
            goals=goals,
            result=result,
            llm_ctx=llm_ctx,
            available_tools_description=atd,
            case_key=case_key,
        )
        extra_fields.update(llm_eval_result)
        if _dm_err is None:
            accuracy = 1.0 if llm_eval_result.get("result", False) else 0.0
        else:
            _force_llm_judge_fail(extra_fields, cfg.llm_eval_fields)

    elif is_semantic_similarity_eval_mode(cfg.eval_mode):
        from benchmarking.evaluation.semantic_similarity import embedding_cosine_similarity

        pred = _resolve_semantic_pred_text(cfg, result)
        ref = _resolve_semantic_ref_text(cfg, case)
        sim, ok = embedding_cosine_similarity(pred, ref)
        threshold = float(cfg.semantic_similarity_threshold or 0.85)
        matched = bool(ok and sim >= threshold)
        extra_fields["semantic_pred_text"] = pred[:500] if pred else ""
        extra_fields["semantic_ref_text"] = ref[:500] if ref else ""
        extra_fields["semantic_similarity"] = round(sim, 4) if ok else None
        extra_fields["semantic_similarity_threshold"] = threshold
        extra_fields["semantic_match"] = matched
        if _dm_err is None:
            accuracy = 1.0 if matched else 0.0

    elif cfg.eval_mode == "Проверка вызова тулзов":
        goal_set = set(case.get("goals", []))
        tool_calls = []
        for msg in reversed(result["history"]):
            if msg["role"] == "assistant":
                full_resp = msg.get("full_response", {})
                if isinstance(full_resp, dict) and "tool_calls" in full_resp:
                    tool_calls = full_resp["tool_calls"]
                    break

        if goal_set == {"None"}:
            is_correct = len(tool_calls) == 0
        else:
            called_names = {
                tc.get("function", {}).get("name")
                for tc in tool_calls
                if isinstance(tc, dict)
            }
            is_correct = bool(goal_set & called_names)

        accuracy = 1.0 if is_correct else 0.0

    elif cfg.eval_mode == "Достигнут блок (block_id)":
        targets = _parse_goal_block_ids(
            case,
            goals,
            dm_stop_default=int(cfg.dm_stop_at_block_id or 0),
        )
        visited = set(_collect_block_ids_from_dm_history(result["history"]))
        for x in result.pop("dm_session_extra_block_ids", None) or []:
            v = _coerce_positive_block_id(x)
            if v is not None:
                visited.add(v)
        goal_hit = bool(targets & visited) if targets else False
        early_stop_hit = _dm_stop_after_block_met(case, cfg, result["history"], goals)
        accuracy = 1.0 if goal_hit else 0.0
        extra_fields["block_goal_met"] = goal_hit
        extra_fields["dm_stop_after_block_met"] = early_stop_hit
        extra_fields["target_block_ids"] = sorted(targets)
        extra_fields["visited_block_ids_dm"] = sorted(visited)

    else:
        if (
            cfg.mode in ("Внешний URL", "Сценарий (Dialog Manager)")
            and cfg.eval_mode == "Сравнить цель с полем из ответа"
        ):
            eval_value = _get_nested_value(result["full_response"], cfg.eval_field_path)
        else:
            eval_value = result["eval_value"]

        eval_context = build_custom_eval_context(
            goals=goals,
            response=result["full_response"],
            eval_value=eval_value,
            history=result["history"],
            context=case_context_raw(case),
        )
        try:
            is_correct = evaluate_custom_eval_code(cfg.custom_eval_code, eval_context)
            accuracy = 1.0 if is_correct else 0.0
        except Exception as e:
            log.warning("Ошибка в критерии оценки: %s", e)
            accuracy = 0.0

    _hist = result.get("history")
    _av_as, _av_us = row_avg_reply_times_from_history(_hist)
    if _av_as is not None:
        extra_fields["avg_assistant_reply_sec"] = round(_av_as, 4)
    if _av_us is not None:
        extra_fields["avg_user_reply_sec"] = round(_av_us, 4)

    if _dm_max_turns_exceeded and _dm_err is None:
        note = (
            "Диалог остановлен по лимиту max_turns (число вызовов form_next_phrase). "
            f"Лимит в конфиге: {int(cfg.max_turns)}."
        )
        if _dm_fnp_calls is not None:
            note += f" Фактически вызовов form_next_phrase: {_dm_fnp_calls}."
        extra_fields["dm_max_turns_exceeded"] = True
        extra_fields["dm_max_turns_limit"] = int(cfg.max_turns)
        if _dm_fnp_calls is not None:
            extra_fields["dm_form_next_phrase_calls"] = int(_dm_fnp_calls)
        extra_fields["dm_max_turns_note"] = note
        accuracy = 0.0
        if is_llm_judge_eval_mode(cfg.eval_mode):
            _force_llm_judge_fail(extra_fields, cfg.llm_eval_fields)
            ro = (extra_fields.get("raw_output") or "").strip()
            extra_fields["raw_output"] = (
                f"{note}\n\n{ro}" if ro else note
            )

    log.info(
        "BENCH timeline wall=%s phase=case_processing_done dialog_id=%s "
        "case_total_monotonic_sec=%.3f (до merge в JSONL)",
        _wall_iso(),
        case_key,
        time.monotonic() - t_case0,
    )

    return merge_case_with_benchmark_output(
        case,
        goals_text=goals,
        result=result,
        cfg=cfg,
        accuracy=accuracy,
        extra_fields=extra_fields,
    )


def _process_single_case_timed(
    case: dict,
    cfg: BenchmarkConfig,
    llm_ctx: LLMRuntimeContext,
    *,
    case_index: int,
    cases_total: int,
) -> dict:
    """Один диалог с замером wall-clock (включая оценку), плюс запись dialog_duration_sec."""
    t0 = time.monotonic()
    ck = str(case.get("dialog_id") or case.get("id") or "?")
    try:
        entry = process_single_case(case, cfg, llm_ctx)
    except Exception as e:
        log.warning(
            "Кейс %s/%s dialog_id=%s: ошибка прогона, диалог считается неверным (accuracy=0): %s",
            case_index,
            cases_total,
            case.get("dialog_id"),
            e,
            exc_info=True,
        )
        entry = benchmark_case_failure_merge(case, cfg, e)
    log.info(
        "BENCH timeline wall=%s phase=case_timed_wrapper_end dialog_id=%s case_index=%s/%s "
        "wall_incl_merge_sec=%.3f",
        _wall_iso(),
        ck,
        case_index,
        cases_total,
        time.monotonic() - t0,
    )
    entry["dialog_duration_sec"] = round(time.monotonic() - t0, 4)
    enrich_benchmark_results_timing_inplace([entry])
    return entry


def _case_copy_for_repeat(case: dict, repeat_index: int, repeats_total: int) -> dict:
    """Копия кейса для параллельного повтора с отдельной сессией DM/API."""
    c = copy.deepcopy(case)
    orig = str(case.get("dialog_id") or case.get("id") or "").strip()
    c["_benchmark_repeat_index"] = repeat_index + 1
    c["_benchmark_repeats_total"] = repeats_total
    if orig:
        c["_benchmark_source_dialog_id"] = orig
    if repeats_total > 1:
        c["dialog_id"] = str(uuid.uuid4())
    return c


def _compact_case_repeat_row(repeat_row: dict) -> dict:
    out: Dict[str, Any] = {
        "repeat_index": repeat_row.get("_benchmark_repeat_index"),
        "accuracy": repeat_row.get("accuracy"),
        "turns": repeat_row.get("turns"),
        "dialog_duration_sec": repeat_row.get("dialog_duration_sec"),
    }
    if repeat_row.get("result") is not None:
        out["result"] = repeat_row.get("result")
    ca = repeat_row.get("criterion_accuracy")
    if isinstance(ca, dict) and ca:
        out["criterion_accuracy"] = ca
    dm_id = repeat_row.get("dm_dialog_id") or repeat_row.get("dialog_id")
    if dm_id is not None:
        out["session_dialog_id"] = dm_id
    for key in (
        "benchmark_run_exception",
        "benchmark_run_exception_type",
        "dm_session_error",
        "raw_output",
        "reason",
    ):
        if repeat_row.get(key):
            out[key] = repeat_row.get(key)
    hist = repeat_row.get("history")
    if isinstance(hist, list) and hist:
        out["history"] = hist
    return out


def aggregate_case_repeat_results(
    case: dict,
    repeats: Sequence[dict],
    cfg: BenchmarkConfig,
) -> dict:
    """Агрегация k параллельных прогонов одного кейса в одну строку results.jsonl."""
    k = len(repeats)
    if k == 0:
        return benchmark_case_failure_merge(
            case, cfg, RuntimeError("Нет результатов повторов кейса")
        )
    if k == 1:
        out = copy.deepcopy(repeats[0])
    else:
        out = copy.deepcopy(repeats[0])

    orig_dialog_id = (
        case.get("dialog_id")
        or case.get("id")
        or repeats[0].get("_benchmark_source_dialog_id")
    )
    if orig_dialog_id is not None and str(orig_dialog_id).strip():
        out["dialog_id"] = str(orig_dialog_id).strip()

    accuracies = [float(r.get("accuracy", 0) or 0) for r in repeats]
    mean_acc = sum(accuracies) / k
    out["accuracy"] = round(mean_acc, 4)
    out["pass_at_least_once"] = any(a >= 1.0 for a in accuracies)
    out["pass_all"] = all(a >= 1.0 for a in accuracies)
    out["repeats_per_case"] = k
    out["repeat_accuracies"] = [round(a, 4) for a in accuracies]
    out["case_repeats"] = [_compact_case_repeat_row(r) for r in repeats]

    if is_llm_judge_eval_mode(cfg.eval_mode):
        scoring = llm_eval_scoring_field_names(parse_llm_eval_fields(cfg.llm_eval_fields))
        crit_acc: Dict[str, float] = {}
        crit_scores: Dict[str, bool] = {}
        for field in scoring:
            vals: List[float] = []
            bools: List[bool] = []
            for row in repeats:
                per = row_criterion_accuracy(row, scoring)
                if field in per:
                    vals.append(float(per[field]))
                    bools.append(float(per[field]) >= 1.0)
            if vals:
                crit_acc[field] = round(sum(vals) / len(vals), 4)
                crit_scores[field] = all(bools)
        if crit_acc:
            out["criterion_accuracy"] = crit_acc
            out["criterion_scores"] = crit_scores
        results_bool = [bool(r.get("result")) for r in repeats]
        out["result_pass_rate"] = round(sum(1 for x in results_bool if x) / k, 4)
        out["result"] = any(results_bool)

    for key in list(out.keys()):
        if key.startswith("_benchmark_"):
            out.pop(key, None)

    return out


def _process_case_with_repeats(
    case: dict,
    cfg: BenchmarkConfig,
    llm_ctx: LLMRuntimeContext,
    *,
    case_index: int,
    cases_total: int,
) -> dict:
    """Один кейс: k параллельных прогонов с опциональным stagger между стартами."""
    k = max(1, int(getattr(cfg, "repeats_per_case", 1) or 1))
    if k == 1:
        return _process_single_case_timed(
            case,
            cfg,
            llm_ctx,
            case_index=case_index,
            cases_total=cases_total,
        )

    stagger = max(0.0, float(getattr(cfg, "repeats_stagger_sec", 1.0) or 0.0))
    source_id = case.get("dialog_id") or case.get("id") or "?"
    t0 = time.monotonic()
    log.info(
        "BENCH timeline wall=%s phase=repeats_begin source_dialog_id=%s case_index=%s/%s "
        "repeats_per_case=%s stagger_sec=%s",
        _wall_iso(),
        source_id,
        case_index,
        cases_total,
        k,
        stagger,
    )

    def _run_one(repeat_index: int) -> dict:
        if stagger > 0 and repeat_index > 0:
            time.sleep(repeat_index * stagger)
        rep_case = _case_copy_for_repeat(case, repeat_index, k)
        return _process_single_case_timed(
            rep_case,
            cfg,
            llm_ctx,
            case_index=case_index,
            cases_total=cases_total,
        )

    repeats: List[Optional[dict]] = [None] * k
    with ThreadPoolExecutor(max_workers=k, thread_name_prefix="bench-rep") as pool:
        futures = {pool.submit(_run_one, i): i for i in range(k)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                repeats[idx] = fut.result()
            except Exception as e:
                log.warning(
                    "Повтор %s/%s для dialog_id=%s: ошибка: %s",
                    idx + 1,
                    k,
                    source_id,
                    e,
                    exc_info=True,
                )
                repeats[idx] = benchmark_case_failure_merge(
                    _case_copy_for_repeat(case, idx, k),
                    cfg,
                    e,
                )

    merged = aggregate_case_repeat_results(
        case, [r for r in repeats if r is not None], cfg
    )
    merged["dialog_duration_sec"] = round(time.monotonic() - t0, 4)
    enrich_benchmark_results_timing_inplace([merged])
    log.info(
        "BENCH timeline wall=%s phase=repeats_done source_dialog_id=%s "
        "mean_accuracy=%s pass_at_least_once=%s wall_sec=%.3f",
        _wall_iso(),
        source_id,
        merged.get("accuracy"),
        merged.get("pass_at_least_once"),
        merged.get("dialog_duration_sec"),
    )
    return merged


def benchmark_case_failure_merge(
    case: dict,
    cfg: BenchmarkConfig,
    exc: Exception,
) -> dict:
    """Запись результата при необработанном исключении в прогоне одного диалога (accuracy=0)."""
    goals = _goals_from_case(case)
    result = {
        "predicted_intents": "",
        "eval_value": "",
        "full_response": {"error": str(exc), "error_type": type(exc).__name__},
        "turns": 0,
        "history": history_from_case(case),
    }
    extra = {
        "benchmark_run_exception": str(exc),
        "benchmark_run_exception_type": type(exc).__name__,
    }
    _hist = result.get("history")
    _av_as, _av_us = row_avg_reply_times_from_history(_hist)
    if _av_as is not None:
        extra["avg_assistant_reply_sec"] = round(_av_as, 4)
    if _av_us is not None:
        extra["avg_user_reply_sec"] = round(_av_us, 4)
    return merge_case_with_benchmark_output(
        case,
        goals_text=goals,
        result=result,
        cfg=cfg,
        accuracy=0.0,
        extra_fields=extra,
    )


def _stamp_benchmark_rows_run_wall(rows: List[dict], t0: float) -> None:
    """Одинаковый wall-clock прогона во всех строках (для JSONL и UI по мере готовности)."""
    elapsed = round(time.monotonic() - t0, 4)
    for row in rows:
        row["benchmark_run_duration_sec"] = elapsed


def run_benchmark_cases(
    test_cases: List[dict],
    cfg: BenchmarkConfig,
    default_model: str,
    stop_check: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int, List[dict]], None]] = None,
) -> List[dict]:
    n = len(test_cases)
    t_run_wall0 = time.monotonic()

    _api_base = (cfg.litellm_api_base or "").strip()
    llm_ctx = LLMRuntimeContext(
        cfg.roles,
        api_base=_api_base if _api_base else None,
        default_model=default_model,
    )
    results: List[dict] = []
    t_prev_case_end: Optional[float] = None
    for i, case in enumerate(test_cases):
        t_loop = time.monotonic()
        gap = None if t_prev_case_end is None else round(t_loop - t_prev_case_end, 4)
        log.info(
            "BENCH timeline wall=%s phase=run_queue dialog_id=%s case_index=%s/%s "
            "gap_since_prev_case_finished_sec=%s | между кейсами: колбэк on_progress "
            "(MinIO/БД), подготовка следующего кейса",
            _wall_iso(),
            case.get("dialog_id"),
            i + 1,
            n,
            gap,
        )
        if stop_check and stop_check():
            break
        entry = _process_case_with_repeats(
            case, cfg, llm_ctx, case_index=i + 1, cases_total=n
        )
        results.append(entry)
        t_prev_case_end = time.monotonic()
        if on_progress:
            t_cb0 = time.monotonic()
            _stamp_benchmark_rows_run_wall(results, t_run_wall0)
            enrich_benchmark_results_timing_inplace(results)
            on_progress(i + 1, n, results)
            log.info(
                "BENCH timeline wall=%s phase=on_progress_done dialog_id=%s case_index=%s/%s "
                "callback_monotonic_sec=%.3f",
                _wall_iso(),
                case.get("dialog_id"),
                i + 1,
                n,
                time.monotonic() - t_cb0,
            )
    _stamp_benchmark_rows_run_wall(results, t_run_wall0)
    enrich_benchmark_results_timing_inplace(results)
    return results


def parse_jsonl_cases(text: str) -> List[dict]:
    test_cases = []
    for line_no, line in enumerate(text.strip().split("\n"), 1):
        if line.strip():
            case = json.loads(line)
            normalize_parsed_case(case)
            merged, err = _normalize_one_benchmark_case(case, f"jsonl:{line_no}")
            if merged is not None:
                case = merged
            elif err:
                log.warning("parse_jsonl_cases: %s", err)
            if "dialog_id" not in case:
                case["dialog_id"] = str(uuid.uuid4())
            test_cases.append(case)
    return test_cases
