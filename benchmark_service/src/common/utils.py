import hashlib
import json
import math
import os
from typing import Any, Collection, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from common.security import REDACT_PLACEHOLDER, normalize_key_name, redact_secrets_for_display
from common.time import DISPLAY_TZ_UTC_PLUS_5, format_datetime_utc_plus_5
from integrations.litellm import LITELLM_API_BASE, LITELLM_API_KEY, get_model_names

load_dotenv()

def _normalize_key_name(name: str) -> str:
    """Совместимый псевдоним для старого приватного helper."""

    return normalize_key_name(name)


def benchmark_result_to_annotation_jsonl_record(row: dict) -> dict:
    """
    Формат одной строки JSONL, как при загрузке «Загрузить готовый JSONL» на странице Разметка:
    ``dialog_id``, ``scenario_id``, ``goals``, ``history`` (только ``role``, ``content``, при необходимости
    ``tool_calls``). Без ``full_response`` и прочих полей ответа внешнего API / оценки бенчмарка.
    При наличии ``context`` он включается (секреты проходят через ``redact_secrets_for_display``).
    """
    did = row.get("dialog_id")
    if did is None:
        did = ""
    sid = row.get("scenario_id")
    if sid is None:
        sid = ""

    goals = row.get("goals")
    if isinstance(goals, str):
        gl: List[str] = [goals] if goals.strip() else [""]
    elif isinstance(goals, list) and goals:
        gl = [str(g) for g in goals]
    else:
        gt = row.get("goals_text")
        if isinstance(gt, str) and gt.strip():
            gl = [gt.strip()]
        else:
            gl = [""]

    history_out: List[dict] = []
    for msg in row.get("history") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        sm: Dict[str, Any] = {"role": role}
        raw_c = msg.get("content", "")
        sm["content"] = raw_c if isinstance(raw_c, str) else str(raw_c or "")

        tool_calls = msg.get("tool_calls")
        fr = msg.get("full_response")
        if not tool_calls and isinstance(fr, dict):
            tool_calls = fr.get("tool_calls")
        if tool_calls:
            sm["tool_calls"] = tool_calls

        history_out.append(sm)

    out: Dict[str, Any] = {
        "dialog_id": did,
        "scenario_id": sid,
        "goals": gl,
        "history": history_out,
    }
    ctx = row.get("context")
    if isinstance(ctx, dict) and ctx:
        out["context"] = redact_secrets_for_display(ctx)
    elif isinstance(ctx, str) and ctx.strip():
        out["context"] = ctx.strip()

    return out


def benchmark_results_to_annotation_jsonl(rows: List[Any]) -> str:
    """Несколько результатов бенчмарка → один текст JSONL (как для Разметки)."""
    lines: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rec = benchmark_result_to_annotation_jsonl_record(row)
        lines.append(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines)


def metric_float(value: Any) -> Optional[float]:
    """Число для метрик и длительностей; отбрасывает bool и нечисловые значения."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


USER_ROLES_REPLY_TIMING = frozenset(
    {
        "user",
        "human",
        "client",
        "customer",
        "end_user",
        "enduser",
        "player",
        "пользователь",
        "клиент",
        "абонент",
    }
)
ASSISTANT_ROLES_REPLY_TIMING = frozenset(
    {
        "assistant",
        "operator",
        "bot",
        "assistant_bot",
        "model",
        "agent",
        "ai",
        "оператор",
        "ассистент",
    }
)

_MESSAGE_TYPE_TO_ROLE = {
    "human": "user",
    "client": "user",
    "customer": "user",
    "end_user": "user",
    "bot": "assistant",
    "ai": "assistant",
    "operator": "operator",
}


def coerce_history_list_for_timing(raw: Any) -> List[Dict[str, Any]]:
    """history как list[dict] или JSON-строка → список реплик."""
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            return []
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def role_for_reply_timing(turn: Dict[str, Any]) -> str:
    """Роль для метрик: role / speaker / message_type (human|bot|…)."""
    r = str(turn.get("role", "")).strip().lower()
    if r:
        return r
    sp = str(turn.get("speaker", "")).strip().lower()
    if sp:
        return sp
    mt = str(turn.get("message_type", "")).strip().lower()
    return _MESSAGE_TYPE_TO_ROLE.get(mt, mt)


def _parse_turn_timestamp(turn: Dict[str, Any]) -> Optional[float]:
    """Unix-время реплики по ts / time / created_at (число или строка)."""
    for k in ("ts", "timestamp", "time", "created_at"):
        v = turn.get(k)
        if v is None:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                continue
            s_iso = s.replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(s_iso)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                pass
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                try:
                    parsed = datetime.strptime(s, fmt)
                    return parsed.replace(tzinfo=timezone.utc).timestamp()
                except ValueError:
                    continue
    return None


def _history_with_inferred_reply_durations(hist: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Копия истории: для реплик без reply_duration_sec проставляет интервал от предыдущей
    по time/created_at/ts (формат разметки и логов).
    """
    out: List[Dict[str, Any]] = []
    for t in hist:
        out.append(dict(t))
    for i in range(1, len(out)):
        if turn_reply_duration_sec(out[i]) is not None:
            continue
        t0 = _parse_turn_timestamp(out[i - 1])
        t1 = _parse_turn_timestamp(out[i])
        if t0 is None or t1 is None or t1 < t0:
            continue
        out[i]["reply_duration_sec"] = round(t1 - t0, 4)
    return out


def history_prepared_for_timing_metrics(raw: Any) -> List[Dict[str, Any]]:
    """История для расчёта метрик: разбор строки + вывод длительностей из меток времени."""
    base = coerce_history_list_for_timing(raw)
    if not base:
        return []
    return _history_with_inferred_reply_durations(base)


def _duration_sec_from_dm_full_response(fr: Any, depth: int = 0) -> Optional[float]:
    """NDA: разбор вложенной структуры full_response Dialog Manager скрыт."""
    if depth > 0 or not isinstance(fr, dict):
        return None
    for key in ("reply_duration_sec", "duration", "latency_sec", "latency_ms"):
        v = fr.get(key)
        if v is None or isinstance(v, bool):
            continue
        f = metric_float(v)
        if f is not None and f >= 0:
            return f / 1000.0 if key == "latency_ms" else f
    return None


def turn_reply_duration_sec(turn: Dict[str, Any]) -> Optional[float]:
    """Длительность ответа на ходе (сек), если указана в известных полях сообщения."""
    for key in ("reply_duration_sec", "latency_sec", "response_time_sec", "duration"):
        f = metric_float(turn.get(key))
        if f is not None and f >= 0:
            return f
    f_ms = metric_float(turn.get("latency_ms"))
    if f_ms is not None and f_ms >= 0:
        return f_ms / 1000.0
    fr = turn.get("full_response")
    d = _duration_sec_from_dm_full_response(fr)
    if d is not None:
        return d
    return None


def _mutate_history_timing_from_dm_and_wall(row: dict) -> None:
    """
    Подставляет reply_duration_sec в history: из full_response (DM), затем остаток от dialog_duration_sec.
    Мутирует row['history'] для стабильного UI/экспорта.
    """
    hist = row.get("history")
    if not isinstance(hist, list):
        return
    for msg in hist:
        if not isinstance(msg, dict):
            continue
        if metric_float(msg.get("reply_duration_sec")) is not None:
            continue
        d = _duration_sec_from_dm_full_response(msg.get("full_response"))
        if d is not None:
            msg["reply_duration_sec"] = round(float(d), 4)
    wall = metric_float(row.get("dialog_duration_sec"))
    if wall is None:
        return
    missing: List[int] = []
    for i, m in enumerate(hist):
        if not isinstance(m, dict):
            continue
        if metric_float(m.get("reply_duration_sec")) is None:
            missing.append(i)
    if not missing:
        return
    existing = sum(
        (metric_float(hist[i].get("reply_duration_sec")) or 0.0)
        for i in range(len(hist))
        if isinstance(hist[i], dict)
    )
    budget = max(0.0, float(wall) - float(existing))
    if budget <= 0:
        return
    share = budget / len(missing)
    for i in missing:
        hist[i]["reply_duration_sec"] = round(share, 4)


def row_avg_reply_times_from_history(
    history: Any,
) -> Tuple[Optional[float], Optional[float]]:
    """
    По списку реплик: среднее время ответа ассистента и пользователя (сек).
    Возвращает (avg_assistant, avg_user) — только по ходам с известной длительностью.
    """
    hist = history_prepared_for_timing_metrics(history)
    if not hist:
        return None, None
    as_vals: List[float] = []
    us_vals: List[float] = []
    for turn in hist:
        role = role_for_reply_timing(turn)
        d = turn_reply_duration_sec(turn)
        if d is None:
            continue
        if role in USER_ROLES_REPLY_TIMING:
            us_vals.append(d)
        elif role in ASSISTANT_ROLES_REPLY_TIMING:
            as_vals.append(d)
    a = sum(as_vals) / len(as_vals) if as_vals else None
    u = sum(us_vals) / len(us_vals) if us_vals else None
    return a, u


def row_dialog_duration_sec_effective(row: dict) -> Optional[float]:
    """
    Длительность диалога для метрик: явное dialog_duration_sec или сумма известных
    задержек по ходам (если воркер старый / поле не сериализовано).
    """
    d = metric_float(row.get("dialog_duration_sec"))
    if d is not None:
        return d
    hist = history_prepared_for_timing_metrics(row.get("history"))
    if not hist:
        return None
    total = 0.0
    n = 0
    for turn in hist:
        t = turn_reply_duration_sec(turn)
        if t is not None:
            total += t
            n += 1
    return total if n else None


_TIMING_ROW_TOP_KEYS = (
    "dialog_duration_sec",
    "benchmark_run_duration_sec",
    "avg_assistant_reply_sec",
    "avg_user_reply_sec",
)


def _normalize_timing_top_level_fields(row: dict) -> None:
    """
    Приводит верхнеуровневые поля времени к float (Postgres/JSON часто отдают строки).
    Пустые строки и нечисла убираем, чтобы metric_float дальше не ломался.
    """
    for k in _TIMING_ROW_TOP_KEYS:
        if k not in row:
            continue
        v = row[k]
        if v is None or isinstance(v, bool):
            row.pop(k, None)
            continue
        if isinstance(v, str) and not str(v).strip():
            row.pop(k, None)
            continue
        f = metric_float(v)
        if f is None or not math.isfinite(f):
            row.pop(k, None)
            continue
        row[k] = round(float(f), 4)


def enrich_benchmark_results_timing_inplace(rows: List[Any]) -> None:
    """
    Дополняет строки результатов метриками времени из ``history`` (как при финализации воркером).
    Идемпотентно: не перезаписывает уже заданные числовые поля.
    """
    for x in rows:
        if not isinstance(x, dict):
            continue
        _normalize_timing_top_level_fields(x)
        _mutate_history_timing_from_dm_and_wall(x)
        if metric_float(x.get("dialog_duration_sec")) is None:
            eff = row_dialog_duration_sec_effective(x)
            if eff is not None:
                x["dialog_duration_sec"] = round(float(eff), 4)
        # После выставления dialog_duration_sec из суммы реплик — второй проход
        # распределяет остаток по ходам без reply_duration_sec.
        _mutate_history_timing_from_dm_and_wall(x)
        ha, hu = row_avg_reply_times_from_history(x.get("history"))
        if metric_float(x.get("avg_assistant_reply_sec")) is None and ha is not None:
            x["avg_assistant_reply_sec"] = round(float(ha), 4)
        if metric_float(x.get("avg_user_reply_sec")) is None and hu is not None:
            x["avg_user_reply_sec"] = round(float(hu), 4)


def benchmark_mean_dialog_durations(rows: List[dict]) -> Optional[float]:
    """Среднее время диалога по строкам (dialog_duration_sec или сумма reply_duration_sec по history)."""
    vals: List[float] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        d = row_dialog_duration_sec_effective(r)
        if d is not None:
            vals.append(d)
    return sum(vals) / len(vals) if vals else None


def benchmark_mean_reply_times(rows: List[dict]) -> Tuple[Optional[float], Optional[float]]:
    """
    Средние ``avg_assistant_reply_sec`` и ``avg_user_reply_sec`` по диалогам.
    Если в строке нет полей — пересчитывает из ``history`` (по ``reply_duration_sec`` и ролям).
    """
    as_t: List[float] = []
    us_t: List[float] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        a = metric_float(r.get("avg_assistant_reply_sec"))
        u = metric_float(r.get("avg_user_reply_sec"))
        if a is None or u is None:
            ha, hu = row_avg_reply_times_from_history(r.get("history"))
            if a is None:
                a = ha
            if u is None:
                u = hu
        if a is not None:
            as_t.append(a)
        if u is not None:
            us_t.append(u)
    ma = sum(as_t) / len(as_t) if as_t else None
    mu = sum(us_t) / len(us_t) if us_t else None
    return ma, mu


def extract_tool_calls_from_jsonl_history_message(msg: Any) -> List[Any]:
    """
    Сообщение ``history`` из готового JSONL: вызовы инструментов могут быть
    на верхнем уровне или внутри ``full_response`` / ``full_response.response``.
    """
    if not isinstance(msg, dict):
        return []

    def _as_list(tc: Any) -> List[Any]:
        if tc is None:
            return []
        if isinstance(tc, list):
            return tc
        if isinstance(tc, dict):
            return [tc]
        return []

    tc = msg.get("tool_calls")
    out = _as_list(tc)
    if out:
        return out

    fr = msg.get("full_response")
    if isinstance(fr, dict):
        out = _as_list(fr.get("tool_calls"))
        if out:
            return out
        resp = fr.get("response")
        if isinstance(resp, dict):
            out = _as_list(resp.get("tool_calls"))
            if out:
                return out
    return []


def tool_calls_for_timeline_ui_display(tool_calls: Any) -> Any:
    """
    Карточка разметки: только поля как у OpenAI (function.name / function.arguments как строка JSON,
    id, type). Аргументы нормализуются: кириллица без \\uXXXX (ensure_ascii=False), снята обёртка
    tool_arguments при единственном ключе, из словарей убран служебный block_id.
    """
    if tool_calls is None:
        return []
    if not isinstance(tool_calls, list):
        return tool_calls
    return [_normalize_tool_call_dict_for_ui(tc) for tc in tool_calls]


def _normalize_arguments_object_for_ui(obj: Any) -> Any:
    """Развернуть единственную обёртку tool_arguments; убрать block_id во вложенных dict."""
    if isinstance(obj, dict):
        if len(obj) == 1 and "tool_arguments" in obj:
            inner = obj["tool_arguments"]
            if isinstance(inner, (dict, list)):
                return _normalize_arguments_object_for_ui(inner)
        return {
            k: _normalize_arguments_object_for_ui(v)
            for k, v in obj.items()
            if k != "block_id"
        }
    if isinstance(obj, list):
        return [_normalize_arguments_object_for_ui(x) for x in obj]
    return obj


def _normalize_tool_arguments_string_for_ui(args_val: Any) -> str:
    """Строка JSON для function.arguments: читаемый Unicode, без лишней обёртки tool_arguments."""
    if args_val is None:
        return "{}"
    if isinstance(args_val, str):
        s = args_val.strip()
        if not s:
            return "{}"
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return args_val
    elif isinstance(args_val, (dict, list)):
        obj = args_val
    else:
        try:
            return json.dumps(args_val, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            return "{}"
    norm = _normalize_arguments_object_for_ui(obj)
    try:
        return json.dumps(norm, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return "{}"


def _normalize_tool_call_dict_for_ui(tc: Any) -> Any:
    if not isinstance(tc, dict):
        return tc
    fn_raw = tc.get("function")
    if not isinstance(fn_raw, dict):
        return {k: v for k, v in tc.items() if k in ("function", "id", "type")}
    args_val = fn_raw.get("arguments", "{}")
    args_out = _normalize_tool_arguments_string_for_ui(args_val)
    return {
        "function": {
            "name": fn_raw.get("name"),
            "arguments": args_out,
        },
        "id": tc.get("id"),
        "type": tc.get("type", "function"),
    }


def format_tool_calls_json_pretty(tool_calls: Any) -> str:
    """Человекочитаемый JSON для отображения tool_calls в UI разметки."""
    if tool_calls is None:
        return "[]"
    try:
        return json.dumps(tool_calls, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return json.dumps(str(tool_calls), ensure_ascii=False, indent=2)


def restrict_variables_snapshot_to_allowlist(
    snapshot: Any,
    allowed_variable_names: Optional[Collection[str]],
) -> Dict[str, Any]:
    """
    Если allowed_variable_names задан и непустой — только эти ключи (имена переменных).
    Если allowlist отсутствует или пустой — возвращает весь snapshot (без ограничения).
    """
    if not snapshot or not isinstance(snapshot, dict):
        return {}
    if not allowed_variable_names:
        return dict(snapshot)
    allow = {str(x).strip() for x in allowed_variable_names if x is not None and str(x).strip()}
    if not allow:
        return dict(snapshot)
    return {str(k): v for k, v in snapshot.items() if str(k) in allow}


def restrict_variables_rows_to_allowlist(
    variables_raw: Any,
    allowed_variable_names: Optional[Collection[str]],
) -> List[dict]:
    """Фильтр строк лога variables по именам (как restrict_variables_snapshot_to_allowlist)."""
    if not variables_raw or not isinstance(variables_raw, list):
        return []
    if not allowed_variable_names:
        return [r for r in variables_raw if isinstance(r, dict)]
    allow = {str(x).strip() for x in allowed_variable_names if x is not None and str(x).strip()}
    if not allow:
        return [r for r in variables_raw if isinstance(r, dict)]
    out: List[dict] = []
    for r in variables_raw:
        if not isinstance(r, dict):
            continue
        vn = r.get("variable_name")
        if vn is not None and str(vn).strip() in allow:
            out.append(r)
    return out


# Один диалог может содержать и вызовы LLM из tech_logs, и синтетику по variables (http_blocks)
TOOL_EVENT_SOURCE_TECH_LOG = "tech_log"
TOOL_EVENT_SOURCE_HTTP_VARIABLES = "http_variables"


def is_tool_event_from_http_variables(msg: Any) -> bool:
    """
    True только для synthetic tool_call из переменных scenario (HTTP-блоки).
    Вызовы из логов платформы в том же диалоге помечены tool_event_source=tech_log или без legacy-маркеров.
    """
    if not isinstance(msg, dict):
        return False
    if msg.get("tool_event_source") == TOOL_EVENT_SOURCE_HTTP_VARIABLES:
        return True
    if msg.get("type") != "tool_call":
        return False
    if msg.get("tool_event_source") == TOOL_EVENT_SOURCE_TECH_LOG:
        return False
    # старые сохранённые таймлайны: только synthetic имел http_tool_variable_names
    return "http_tool_variable_names" in msg


def skip_timeline_tech_log_tool_when_http_variables_ui(msg: Any, *, http_variables_ui: bool) -> bool:
    """
    Когда заданы имена переменных для HTTP-блоков (tool_name / tool_arguments),
    в хронологии скрываем события инструментов из tech_logs (NDA: формат логов платформы скрыт),
    чтобы не смешивать их с synthetic по variables.
    Без имён переменных такие строки показываются (см. цикл разметки).
    """
    if not http_variables_ui:
        return False
    if not isinstance(msg, dict):
        return False
    if msg.get("tool_event_source") != TOOL_EVENT_SOURCE_TECH_LOG:
        return False
    return msg.get("type") in ("tool_call", "tool_schema", "tool_message")


def tech_log_openai_schema_as_markup_tool_calls(
    tool_schema: Any,
    *,
    tool_variable: Optional[str] = None,
) -> List[dict]:
    """
    OpenAI-определение инструмента из tech_logs → тот же вид элементов, что в timeline для tool_call
    (function.name, function.arguments как JSON-строка, id, type).
    """
    if not isinstance(tool_schema, dict):
        return []
    fn = tool_schema.get("function")
    if not isinstance(fn, dict):
        return []
    raw_name = fn.get("name")
    if not raw_name:
        return []
    name = str(raw_name)
    body = {k: v for k, v in fn.items() if k != "name"}
    args_str = json.dumps(body, ensure_ascii=False) if body else "{}"
    stable = f"{tool_variable or ''}\0{name}\0{args_str}"
    digest = hashlib.md5(stable.encode("utf-8")).hexdigest()[:24]
    return [
        {
            "function": {
                "name": name,
                "arguments": args_str,
            },
            "id": f"call_{digest}",
            "type": "function",
        }
    ]


def markup_timeline_row_icon(msg: dict) -> str:
    """Иконка строки хронологии на страницах разметки: 👤 пользователь, 💼 ассистент, 🔧 tool."""
    role = (msg.get("role") or "").strip().lower()
    if role in ("user", "human", "client"):
        return "👤"
    if role in ("tool",):
        return "🔧"
    return "💼"
