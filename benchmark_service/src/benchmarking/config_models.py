"""Модели конфигурации бенчмарка и LiteLLM runtime."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from benchmarking.parsing.cases import normalize_parsed_case

log = logging.getLogger(__name__)

BENCHMARK_HTTP_READ_TIMEOUT_SEC = 300.0
BENCHMARK_HTTP_CONNECT_TIMEOUT_SEC = 10.0
BENCHMARK_HTTP_TIMEOUT: Tuple[float, float] = (
    BENCHMARK_HTTP_CONNECT_TIMEOUT_SEC,
    BENCHMARK_HTTP_READ_TIMEOUT_SEC,
)

@dataclass
class RoleLLMConfig:
    model: str = ""
    api_key: str = ""
    params_json: str = "{}"


SEMANTIC_SIMILARITY_EVAL_MODE = "Семантическое сходство"
_LEGACY_SEMANTIC_SIMILARITY_EVAL_MODE = "Семантическое сходство (эмбеддинги)"

LLM_JUDGE_EVAL_MODE = "LLM-судья"
_LEGACY_LLM_JUDGE_EVAL_MODE = "Оценка всего диалога через LLM"


def is_semantic_similarity_eval_mode(eval_mode: str) -> bool:
    return eval_mode in (
        SEMANTIC_SIMILARITY_EVAL_MODE,
        _LEGACY_SEMANTIC_SIMILARITY_EVAL_MODE,
    )


def is_llm_judge_eval_mode(eval_mode: str) -> bool:
    return eval_mode in (LLM_JUDGE_EVAL_MODE, _LEGACY_LLM_JUDGE_EVAL_MODE)


LLM_EVAL_META_FIELD_NAMES = frozenset(
    {"reason", "details", "comment", "explanation", "note"}
)


def parse_llm_eval_fields(raw: str) -> List[str]:
    return [f.strip() for f in (raw or "").split(",") if f.strip()]


def llm_eval_scoring_field_names(expected_fields: List[str]) -> List[str]:
    """Поля JSON-ответа судьи, по которым считается бинарная точность (0/1)."""
    return [f for f in expected_fields if f not in LLM_EVAL_META_FIELD_NAMES]


def coerce_llm_criterion_passed(value: Any) -> Optional[bool]:
    """Разбор значения критерия: 1/0, true/false, dict с passed."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1 or value == 1.0:
            return True
        if value == 0 or value == 0.0:
            return False
    if isinstance(value, dict) and "passed" in value:
        return bool(value.get("passed"))
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("1", "true", "yes", "да", "pass", "passed"):
            return True
        if s in ("0", "false", "no", "нет", "fail", "failed"):
            return False
    return None


def derive_criterion_scores(
    parsed: dict, scoring_fields: List[str]
) -> Dict[str, bool]:
    scores: Dict[str, bool] = {}
    for field in scoring_fields:
        passed = coerce_llm_criterion_passed(parsed.get(field))
        if passed is not None:
            scores[field] = passed
        elif field == "result" and parsed.get("result") is not None:
            scores[field] = bool(parsed.get("result"))
    return scores


def row_criterion_accuracy(row: dict, scoring_fields: List[str]) -> Dict[str, float]:
    """Бинарная точность по критериям для одной строки результатов."""
    ca = row.get("criterion_accuracy")
    if isinstance(ca, dict) and ca:
        return {
            f: float(ca[f])
            for f in scoring_fields
            if f in ca and ca[f] is not None
        }
    out: Dict[str, float] = {}
    for field in scoring_fields:
        passed = coerce_llm_criterion_passed(row.get(field))
        if passed is not None:
            out[field] = 1.0 if passed else 0.0
        elif field == "result" and "result" in row:
            out[field] = 1.0 if row.get("result") else 0.0
    return out


def benchmark_criterion_accuracy_summary(
    results: Sequence[dict],
    llm_eval_fields: str = "",
) -> Dict[str, float]:
    """Средняя точность по каждому критерию LLM-судьи."""
    scoring = llm_eval_scoring_field_names(parse_llm_eval_fields(llm_eval_fields))
    if not scoring:
        return {}
    sums = {f: 0.0 for f in scoring}
    counts = {f: 0 for f in scoring}
    for row in results:
        if not isinstance(row, dict):
            continue
        per_row = row_criterion_accuracy(row, scoring)
        for field, acc in per_row.items():
            sums[field] += acc
            counts[field] += 1
    return {
        f: (sums[f] / counts[f] if counts[f] else 0.0) for f in scoring
    }


def row_mean_criterion_score(
    row: dict, scoring_fields: List[str]
) -> Optional[float]:
    """Средний балл по критериям для одного диалога (0..1)."""
    per_row = row_criterion_accuracy(row, scoring_fields)
    if not per_row:
        return None
    return sum(per_row.values()) / len(per_row)


def benchmark_mean_criterion_score(
    results: Sequence[dict],
    llm_eval_fields: str = "",
) -> Optional[float]:
    """Средний балл по критериям по всем диалогам (среднее per-dialog mean)."""
    scoring = llm_eval_scoring_field_names(parse_llm_eval_fields(llm_eval_fields))
    if not scoring:
        return None
    scores: List[float] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        m = row_mean_criterion_score(row, scoring)
        if m is not None:
            scores.append(m)
    if not scores:
        return None
    return sum(scores) / len(scores)


def _force_llm_judge_fail(extra_fields: dict, llm_eval_fields: str) -> None:
    extra_fields["result"] = False
    scoring = llm_eval_scoring_field_names(parse_llm_eval_fields(llm_eval_fields))
    extra_fields["criterion_scores"] = {f: False for f in scoring}
    extra_fields["criterion_accuracy"] = {f: 0.0 for f in scoring}


@dataclass
class BenchmarkConfig:
    mode: str = "LLM (через LiteLLM)"
    assistant_url: str = ""
    user_message_key: str = "message"
    response_field_path: str = "response"
    assistant_prompt: str = ""
    parse_json_response: bool = False
    use_tools: bool = False
    assistant_tools: str = ""
    user_prompt: str = ""
    max_turns: int = 5
    llm_delay: float = 0.0
    # Число параллельных прогонов одного dialog_id (mean@k для accuracy).
    repeats_per_case: int = 1
    # Пауза перед стартом i-го повтора: i * repeats_stagger_sec (0 = все сразу).
    repeats_stagger_sec: float = 1.0
    # Оставлено для совместимости с сохранёнными конфигами; раннер всегда обрабатывает кейсы по очереди.
    max_parallel_dialogs: int = 1
    eval_mode: str = "Сравнить цель с полем из ответа"
    eval_field_path: str = "response"
    custom_eval_code: str = ""
    evaluate_existing_only: bool = True
    llm_eval_prompt: str = ""
    llm_eval_fields: str = "result,reason"
    semantic_pred_field_path: str = "theme"
    semantic_ref_field_path: str = "subtopic"
    semantic_similarity_threshold: float = 0.85
    exit_when_condition_met: bool = False
    # Внешний URL: JSON {"ключ_в_теле_api": "ключ_в_context", ...}. Пустая строка — весь context в теле как раньше.
    external_context_field_map_json: str = ""
    # Имя поля в JSON тела POST для идентификатора диалога (значение — dialog_id из кейса или новый UUID).
    external_session_id_field: str = "id"
    # Внешний URL: True — в теле API всегда новый UUID на каждый прогон кейса; False — id из поля dialog_id кейса (разметка), иначе UUID.
    external_unique_session_id: bool = False
    # Имена полей в теле POST, для которых строка из context приводится к int (через запятую). Пусто — не приводить.
    external_coerce_int_fields_csv: str = "flat_id"
    # Сценарий (Dialog Manager)
    dm_base_url: str = ""
    dm_start_block_id: int = 0
    # 0 = не ограничивать. Иначе при первом ответе form_next_phrase, где среди block_data.id
    # встретилось это значение: сразу POST end_dialog на DM, обход сценария прекращается
    # (симулятор пользователя больше не вызывается, следующий form_next_phrase не идёт).
    dm_stop_at_block_id: int = 0
    dm_first_user_phrase: str = "*начало диалога*"
    dm_add_data_json: str = "{}"
    # True: в истории для симулятора/оценки первая реплика — оператор; стартовая фраза всё равно уходит в DM API.
    dm_scenario_speaks_first: bool = False
    # LLM / Внешний URL: при пустой history в кейсе — первая реплика от ассистента (иначе от пользователя).
    assistant_speaks_first: bool = False
    # ID сценария на уровне запуска (внешний JSON); попадает в results.jsonl.
    scenario_id: str = ""
    # База OpenAI-совместимого API (LiteLLM proxy): в очередь кладётся из UI/API, воркер берёт её в приоритете над своим env.
    litellm_api_base: str = ""
    roles: Dict[str, RoleLLMConfig] = field(default_factory=dict)


def _config_dm_start_block_id(d: dict) -> int:
    """
    Стартовый блок DM для create_dialog: поле ``dm_start_block_id`` или синоним ``start_block_id``
    (в т.ч. в JSON-шаблонах).

    Приоритет: положительный ``dm_start_block_id``; иначе положительный ``start_block_id``;
    иначе ``dm_start_block_id`` (включая 0); иначе ``start_block_id``; иначе 0.
    """

    def _one(key: str) -> Optional[int]:
        if key not in d:
            return None
        try:
            return int(d[key])
        except (TypeError, ValueError):
            return None

    dm = _one("dm_start_block_id")
    sb = _one("start_block_id")
    if dm is not None and dm > 0:
        return dm
    if sb is not None and sb > 0:
        return sb
    if dm is not None:
        return dm
    if sb is not None:
        return sb
    return 0


def benchmark_config_to_dict(cfg: BenchmarkConfig) -> dict:
    d = asdict(cfg)
    d["roles"] = {k: asdict(v) for k, v in cfg.roles.items()}
    d["start_block_id"] = int(cfg.dm_start_block_id)
    return d


def _config_optional_str(d: dict, key: str, default: str = "") -> str:
    """Строковые поля из JSON/Postgres: ключ с явным null не должен давать None в dataclass."""
    v = d.get(key, default)
    if v is None:
        return default
    return str(v)


def _config_litellm_api_base(d: dict) -> str:
    v = _config_optional_str(d, "litellm_api_base", "").strip()
    if v:
        return v
    em = d.get("enqueue_meta")
    if isinstance(em, dict):
        v2 = _config_optional_str(em, "litellm_api_base_env", "").strip()
        if v2:
            return v2
    return ""


def benchmark_config_from_dict(d: dict) -> BenchmarkConfig:
    roles_raw = d.get("roles") or {}
    roles = {
        k: RoleLLMConfig(**v) if isinstance(v, dict) else RoleLLMConfig()
        for k, v in roles_raw.items()
    }
    for r in ("assistant", "user", "evaluator"):
        roles.setdefault(r, RoleLLMConfig())
    return BenchmarkConfig(
        mode=d.get("mode", "LLM (через LiteLLM)"),
        assistant_url=d.get("assistant_url", ""),
        user_message_key=d.get("user_message_key", "message"),
        response_field_path=d.get("response_field_path", "response"),
        assistant_prompt=_config_optional_str(d, "assistant_prompt", ""),
        parse_json_response=bool(d.get("parse_json_response", False)),
        use_tools=bool(d.get("use_tools", False)),
        assistant_tools=_config_optional_str(d, "assistant_tools", ""),
        user_prompt=_config_optional_str(d, "user_prompt", ""),
        max_turns=int(d.get("max_turns", 5)),
        llm_delay=float(d.get("llm_delay", 0.0)),
        repeats_per_case=max(1, int(d.get("repeats_per_case", 1) or 1)),
        repeats_stagger_sec=max(0.0, float(d.get("repeats_stagger_sec", 1.0) or 0.0)),
        max_parallel_dialogs=1,
        eval_mode=d.get("eval_mode", "Сравнить цель с полем из ответа"),
        eval_field_path=d.get("eval_field_path", "response"),
        custom_eval_code=d.get("custom_eval_code", ""),
        # По умолчанию False: прогон в DM/CМ без сохранённой истории; True только если явно в конфиге.
        evaluate_existing_only=bool(d.get("evaluate_existing_only", False)),
        llm_eval_prompt=_config_optional_str(d, "llm_eval_prompt", ""),
        llm_eval_fields=d.get("llm_eval_fields", "result,reason"),
        semantic_pred_field_path=_config_optional_str(
            d, "semantic_pred_field_path", "theme"
        ),
        semantic_ref_field_path=_config_optional_str(
            d, "semantic_ref_field_path", "subtopic"
        ),
        semantic_similarity_threshold=float(
            d.get("semantic_similarity_threshold", 0.85) or 0.85
        ),
        exit_when_condition_met=bool(d.get("exit_when_condition_met", False)),
        external_context_field_map_json=d.get("external_context_field_map_json", ""),
        external_session_id_field=d.get("external_session_id_field", "id"),
        external_unique_session_id=bool(d.get("external_unique_session_id", False)),
        external_coerce_int_fields_csv=d.get("external_coerce_int_fields_csv", "flat_id"),
        dm_base_url=d.get("dm_base_url", ""),
        dm_start_block_id=_config_dm_start_block_id(d),
        dm_stop_at_block_id=_coerce_positive_block_id(d.get("dm_stop_at_block_id", 0))
        or 0,
        dm_first_user_phrase=_config_optional_str(
            d, "dm_first_user_phrase", "*начало диалога*"
        ),
        dm_add_data_json=_config_optional_str(d, "dm_add_data_json", "{}"),
        dm_scenario_speaks_first=bool(d.get("dm_scenario_speaks_first", False)),
        assistant_speaks_first=bool(d.get("assistant_speaks_first", False)),
        scenario_id=_config_optional_str(d, "scenario_id", ""),
        litellm_api_base=_config_litellm_api_base(d),
        roles=roles,
    )


class LLMRuntimeContext:
    """Параметры LiteLLM по ролям (вместо st.session_state)."""

    # Из params_json не мержим параметры, которые LiteLLM трактует как отдельный system /
    # prompt management / сырые messages — иначе в трейсе и иногда в запросе остаётся
    # буквальный «{user_prompt}», а системный промпт из кода игнорируется.
    _PARAMS_JSON_SKIP_KEYS = frozenset(
        {
            "system",
            "system_message",
            "messages",
            "prompt",
            "prompt_id",
            "prompt_variables",
            "prompt_label",
            "prompt_version",
            "initial_prompt_value",
            "final_prompt_value",
            "roles",
            "bos_token",
            "eos_token",
        }
    )

    def __init__(
        self,
        roles: Dict[str, RoleLLMConfig],
        api_base: Optional[str] = None,
        default_model: str = "openai/gpt-4o-mini",
    ):
        self.roles = roles
        self.api_base = api_base or os.getenv("LITELLM_API_BASE")
        self.default_model = default_model

    def litellm_kwargs(self, role: str) -> dict:
        rc = self.roles.get(role) or RoleLLMConfig()
        model = (rc.model or "").strip() or self.default_model
        api_key = (rc.api_key or "").strip() or os.getenv("LITELLM_API_KEY")
        kwargs = {
            "model": model,
            "api_key": api_key or None,
            "api_base": self.api_base or None,
            "extra_body": {"cache": {"no-cache": True}},
            "timeout": BENCHMARK_HTTP_READ_TIMEOUT_SEC,
        }
        user_json_str = rc.params_json or "{}"
        if isinstance(user_json_str, str) and user_json_str.strip():
            try:
                model_params = json.loads(user_json_str.strip())
            except json.JSONDecodeError:
                model_params = {}
            if isinstance(model_params, dict):
                for k, v in model_params.items():
                    if k in self._PARAMS_JSON_SKIP_KEYS:
                        log.warning(
                            "Роль LiteLLM %r: ключ %r из params_json пропущен — "
                            "системный промпт и сообщения задаёт только бенчмарк.",
                            role,
                            k,
                        )
                        continue
                    kwargs[k] = v
        return kwargs
