"""Парсинг и нормализация benchmark-кейсов."""

from __future__ import annotations

import json
import re
import unicodedata
import uuid
from typing import Any, Dict, List, Optional, Tuple

class _EmptyMissing(dict):
    """Для str.format_map: отсутствующий ключ (нет в context у кейса) → пустая строка."""

    def __missing__(self, key: str) -> str:
        return ""


def normalize_parsed_case(case: dict) -> dict:
    """
    После json.loads строки JSONL: если context — строка с JSON-объектом внутри, распарсить в dict.
    Не исправляет битый JSON в файле (когда кавычки внутри строки не экранированы).
    """
    ctx = case.get("context")
    if isinstance(ctx, str):
        s = ctx.strip()
        if s:
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    case["context"] = parsed
            except json.JSONDecodeError:
                pass
    return case


def _normalize_one_benchmark_case(case: dict, err_label: str) -> Tuple[Optional[dict], Optional[str]]:
    """err_label — например «Кейс 2» или «Строка 5» для текста ошибки."""
    goals = case.get("goals")
    if goals is None:
        goals_list: List[str] = []
    elif isinstance(goals, str):
        s = goals.strip()
        goals_list = [s] if s else []
    elif isinstance(goals, list):
        goals_list = []
        for g in goals:
            if g is None:
                continue
            s = str(g).strip()
            if s:
                goals_list.append(s)
    else:
        return None, (
            f"{err_label}: «goals» должен быть строкой, массивом строк, null или опущен (пустая цель допустима)."
        )

    ctx = case.get("context", {})
    if ctx is None:
        ctx = {}
    if not isinstance(ctx, dict):
        return None, f"{err_label}: «context» должен быть объектом JSON (или опущен)."

    merged = dict(case)
    merged["goals"] = goals_list
    merged["context"] = dict(ctx)
    normalize_parsed_case(merged)
    if not merged.get("dialog_id"):
        merged["dialog_id"] = str(uuid.uuid4())
    return merged, None


def parse_benchmark_cases_json_text(text: str) -> Tuple[List[dict], Optional[str]]:
    """
    Один JSON-файл с кейсами для бенчмарка: либо один объект, либо массив объектов.
    У каждого кейса поле goals — строка, список строк, пустое или опущено (нормализуется в ``[]``);
    context — объект JSON или опущен.
    dialog_id сохраняется, если задан; иначе генерируется UUID.
    """
    stripped = (text or "").strip()
    if not stripped:
        return [], "Файл пуст."
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        return [], f"Невалидный JSON: {e}"

    if isinstance(data, dict):
        items: List[dict] = [data]
    elif isinstance(data, list):
        if not data:
            return [], "Массив кейсов пуст."
        items = []
        for i, el in enumerate(data):
            if not isinstance(el, dict):
                return [], f"Элемент {i + 1}: ожидается объект, не {type(el).__name__}."
            items.append(el)
    else:
        return [], "Ожидается JSON-объект кейса или JSON-массив объектов."

    out: List[dict] = []
    for i, case in enumerate(items):
        merged, err = _normalize_one_benchmark_case(case, f"Кейс {i + 1}")
        if err:
            return [], err
        out.append(merged)
    return out, None


def parse_benchmark_cases_jsonl_text(text: str) -> Tuple[List[dict], Optional[str]]:
    """
    JSONL: каждая непустая строка — один JSON-объект кейса (как при загрузке .jsonl в бенчмарк).
    Те же требования к полям, что и для JSON-файла.
    """
    entries: List[Tuple[str, dict]] = []
    for line_no, line in enumerate((text or "").splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            return [], f"Строка {line_no}: невалидный JSON: {e}"
        if not isinstance(obj, dict):
            return [], (
                f"Строка {line_no}: ожидается объект {{...}}, не {type(obj).__name__}."
            )
        entries.append((f"Строка {line_no}", obj))
    if not entries:
        return [], "Нет непустых строк с JSON-объектами."
    out: List[dict] = []
    for label, case in entries:
        merged, err = _normalize_one_benchmark_case(case, label)
        if err:
            return [], err
        out.append(merged)
    return out, None


def _json_equal_for_merge(a: Any, b: Any) -> bool:
    try:
        return json.dumps(a, sort_keys=True, separators=(",", ":")) == json.dumps(
            b, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError):
        return False


def common_context_from_cases(cases: List[dict]) -> dict:
    """
    Поля context, которые есть у каждого кейса и совпадают по значению — выносятся в общий блок формы.
    """
    if not cases:
        return {}
    ctxs: List[dict] = []
    for c in cases:
        ctx = c.get("context")
        ctxs.append(ctx if isinstance(ctx, dict) else {})
    common: dict = {}
    for k in set(ctxs[0].keys()):
        if not all(k in cx for cx in ctxs):
            continue
        v0 = ctxs[0][k]
        if all(_json_equal_for_merge(cx[k], v0) for cx in ctxs):
            common[k] = v0
    return common


def goals_text_from_case(case: dict) -> str:
    g = case.get("goals")
    if isinstance(g, str):
        return g.strip()
    if isinstance(g, list) and g:
        return " | ".join(str(x).strip() for x in g)
    return ""


def _context_without_stop_keys(ctx: dict) -> dict:
    out = dict(ctx)
    out.pop("dm_stop_at_block_id", None)
    out.pop("stop_at_block_id", None)
    return out


def dm_stop_block_id_from_case(case: dict) -> int:
    for key in ("dm_stop_at_block_id", "stop_at_block_id"):
        v = case.get(key)
        try:
            if v is not None and int(v) > 0:
                return int(v)
        except (TypeError, ValueError):
            pass
    ctx = case.get("context")
    if isinstance(ctx, dict):
        for key in ("dm_stop_at_block_id", "stop_at_block_id"):
            v = ctx.get(key)
            try:
                if v is not None and int(v) > 0:
                    return int(v)
            except (TypeError, ValueError):
                pass
    return 0


def cases_to_assistant_gen_form_rows(cases: List[dict]) -> Tuple[List[dict], dict]:
    """
    Сжимает список кейсов в строки формы «генерация с нуля»: одинаковые (цель + персональный context + n)
    схлопываются в одну строку с полем «Диалогов» = число таких кейсов.
    Возвращает (rows, common_context).
    """
    if not cases:
        return [{"text": "", "n": 5, "context": {}}], {}
    common = common_context_from_cases(cases)
    rows_map: Dict[tuple, dict] = {}
    order: List[tuple] = []
    for c in cases:
        text = goals_text_from_case(c)
        ctx = c.get("context") if isinstance(c.get("context"), dict) else {}
        row_ctx = {k: v for k, v in ctx.items() if k not in common}
        key = (
            text,
            json.dumps(row_ctx, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
        if key not in rows_map:
            rows_map[key] = {"text": text, "n": 0, "context": dict(row_ctx)}
            order.append(key)
        rows_map[key]["n"] += 1
    rows = [rows_map[k] for k in order]
    for r in rows:
        r["n"] = max(1, min(500, int(r["n"])))
    return rows, common


def cases_to_dm_bench_gen_form_rows(cases: List[dict]) -> Tuple[List[dict], dict]:
    """
    То же для бенчмарка сценария: в строке формы ещё stop_block_id (из корня/context кейса).
    Поля остановки в персональном context не дублируются — только в числе «Блок цели».
    """
    if not cases:
        return [{"text": "", "stop_block_id": 0, "n": 5, "context": {}}], {}
    pseudo = [
        {"context": _context_without_stop_keys(case_context_raw(c))} for c in cases
    ]
    common = common_context_from_cases(pseudo)
    rows_map: Dict[tuple, dict] = {}
    order: List[tuple] = []
    for c in cases:
        text = goals_text_from_case(c)
        stop = dm_stop_block_id_from_case(c)
        ctx = _context_without_stop_keys(case_context_raw(c))
        row_ctx = {k: v for k, v in ctx.items() if k not in common}
        key = (
            text,
            stop,
            json.dumps(row_ctx, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
        if key not in rows_map:
            rows_map[key] = {
                "text": text,
                "stop_block_id": stop,
                "n": 0,
                "context": dict(row_ctx),
            }
            order.append(key)
        rows_map[key]["n"] += 1
    rows = [rows_map[k] for k in order]
    for r in rows:
        r["n"] = max(1, min(500, int(r["n"])))
    return rows, common


def case_context_raw(case: dict) -> dict:
    """Сырой context из кейса (только dict)."""
    ctx = case.get("context")
    return ctx if isinstance(ctx, dict) else {}


def effective_case_context_dict(case: dict) -> dict:
    """
    Контекст кейса как dict. Если ``context`` — JSON в строке (в т.ч. после json.loads кейса),
    парсим без обязательной мутации ``case`` (дублирует логику ``normalize_parsed_case`` для надёжности).
    """
    ctx = case.get("context")
    if isinstance(ctx, dict):
        return dict(ctx)
    if isinstance(ctx, str):
        s = ctx.strip()
        if s:
            try:
                p = json.loads(s)
                if isinstance(p, dict):
                    return p
            except json.JSONDecodeError:
                pass
    return {}


def _is_literal_user_prompt_placeholder(s: str) -> bool:
    """
    Строка-плейсхолдер из шаблона конфига (не текст персоны).
    Если такое значение попало в кейс, str.format подставит его как значение {user_prompt}
    и в промпте останется буквальный «{user_prompt}».
    Учитываем NFKC и пробелы (копипаст из Word / невидимые символы).
    """
    if not (s or "").strip():
        return False
    t = unicodedata.normalize("NFKC", str(s)).strip()
    t = " ".join(t.split())
    return t in ("{user_prompt}", "{{user_prompt}}")


def _persona_text_for_user_simulator(case: dict, ctx_flat: dict) -> str:
    """
    Текст системного промпта симулятора из ``context.user_prompt`` или корня кейса.
    ``ctx_flat`` — результат :func:`case_context_as_str_dict` (запасной источник).
    Литеральные плейсхолдеры ``{user_prompt}`` пропускаются — берётся следующий кандидат.
    """
    ctx_eff = effective_case_context_dict(case)
    for cand in (
        ctx_eff.get("user_prompt"),
        case.get("user_prompt"),
        ctx_flat.get("user_prompt"),
        ctx_eff.get("resident_prompt"),
        ctx_eff.get("persona"),
        ctx_eff.get("simulator_system_prompt"),
        ctx_eff.get("user_role_prompt"),
    ):
        if cand is None or isinstance(cand, (dict, list)):
            continue
        s = str(cand).strip()
        if not s or _is_literal_user_prompt_placeholder(s):
            continue
        return s
    return ""


def _normalize_history_for_eval(raw: Any) -> List[Dict[str, Any]]:
    """Нормализует history из результата прогона для оценки (list[dict] или JSON-строка)."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return _coerce_history_list(raw)


def _coerce_history_list(raw: Any) -> List[Dict[str, Any]]:
    """Превращает поле кейса в список объектов реплик (или [])."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
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


def history_from_case(case: dict) -> List[Dict[str, Any]]:
    """
    История диалога для прогона и LLM-оценки.

    Берётся из верхнего уровня кейса (``history`` / ``messages``) или из ``context`` —
    на практике встречаются оба варианта, а также JSON-массив в виде строки.
    """
    for key in ("history", "messages"):
        h = _coerce_history_list(case.get(key))
        if h:
            return h
    ctx = case.get("context")
    if isinstance(ctx, dict):
        for key in ("history", "messages"):
            h = _coerce_history_list(ctx.get(key))
            if h:
                return h
    return []


_CASE_TEMPLATE_SKIP_TOP = frozenset(
    {
        "context",
        "history",
        "messages",
        "full_response",
        "goals",
        "goals_text",
    }
)


def case_context_as_str_dict(case: dict) -> dict:
    """
    Плоские строковые значения для подстановки в {placeholders}:
    поля ``context`` кейса плюс скаляры с верхнего уровня (как в JSONL без вложения в context).
    Ключи из ``context`` не перекрываются полями с верхнего уровня.
    """
    out: dict = {}
    for k, v in case_context_raw(case).items():
        if not isinstance(k, str):
            continue
        if isinstance(v, (dict, list)):
            try:
                out[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                out[k] = str(v)
        elif v is None:
            out[k] = ""
        else:
            out[k] = str(v)
    for k, v in case.items():
        if k in _CASE_TEMPLATE_SKIP_TOP or k in out:
            continue
        if not isinstance(k, str):
            continue
        if isinstance(v, (dict, list)):
            continue
        if v is None:
            continue
        out[k] = str(v)
    return out


def case_template_strings_for_eval(case: dict) -> dict:
    """
    Строковые плейсхолдеры для промпта LLM-оценщика: то же, что :func:`case_context_as_str_dict`.
    """
    return case_context_as_str_dict(case)


def merge_template_vars(context_strs: dict, **kwargs: Any) -> dict:
    """Поля из kwargs перекрывают одноимённые ключи из context."""
    out = dict(context_strs)
    for k, v in kwargs.items():
        out[k] = v
    return out


_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def format_prompt_template(template: str, variables: dict) -> str:
    """
    Подстановка плейсхолдеров ``{name}`` в шаблоне.

    Примеры JSON в тексте (``{"result": true}``) не интерпретируются как плейсхолдеры —
    подставляются только идентификаторы вида ``{goals}``, ``{history}``, ключи из ``context``.
    """
    if not template:
        return template
    missing = _EmptyMissing(**variables)

    def _replace(match: re.Match[str]) -> str:
        return missing[match.group(1)]

    return _PLACEHOLDER_RE.sub(_replace, template)


def _coerce_user_prompt_field_template(raw: Any) -> str:
    """
    Если в конфиге указали ``{{user_prompt}}`` (как экранирование для str.format), при пустом
    значении плейсхолдера в трейсе LiteLLM остаётся буквальный ``{user_prompt}``. Для шаблона
    «только подставить персону из кейса» нужна одна пара скобок: ``{user_prompt}``.
    """
    s = "" if raw is None else str(raw)
    if s.strip() == "{{user_prompt}}":
        return "{user_prompt}"
    return s


def resolve_case_prompt_templates(
    case: dict,
    cfg: "BenchmarkConfig",
    goals: str,
    *,
    history_text: str = "",
    available_tools_description: str = "",
) -> tuple:
    """
    Промпты ассистента и пользователя с подстановкой context + goals + history + …
    Возвращает (assistant_rendered, user_rendered, eval_vars).
    """
    ctx = case_context_as_str_dict(case)
    persona = _persona_text_for_user_simulator(case, ctx)
    if (
        "{user_prompt}" in (cfg.user_prompt or "")
        and not (persona or "").strip()
    ):
        log.warning(
            "Кейс dialog_id=%s: в шаблоне симулятора есть {user_prompt}, но нет "
            "текста персоны (ожидается context.user_prompt или корневой user_prompt в кейсе).",
            case.get("dialog_id"),
        )
    base_ap = merge_template_vars(
        ctx,
        goals=goals,
        history=history_text,
        user_prompt="",
        assistant_prompt="",
        available_tools=available_tools_description,
    )
    ap_raw = cfg.assistant_prompt if cfg.mode == "LLM (через LiteLLM)" else ""
    ap = format_prompt_template(ap_raw, base_ap) if ap_raw else ""
    # user_prompt — фактический текст персоны из кейса (не шаблон из cfg), иначе {user_prompt} не раскрывается.
    base_up = merge_template_vars(
        ctx,
        goals=goals,
        history=history_text,
        assistant_prompt=ap,
        available_tools=available_tools_description,
        user_prompt=persona,
    )
    tmpl_u = _coerce_user_prompt_field_template(getattr(cfg, "user_prompt", "") or "")
    if not tmpl_u.strip():
        up = persona or ""
    else:
        try:
            up = format_prompt_template(tmpl_u, base_up)
        except (ValueError, KeyError) as ex:
            log.warning(
                "Кейс dialog_id=%s: ошибка подстановки шаблона симулятора (%s), "
                "используем текст персоны без шаблона.",
                case.get("dialog_id"),
                ex,
            )
            up = persona or ""
        if (
            persona
            and not _is_literal_user_prompt_placeholder(persona)
            and _is_literal_user_prompt_placeholder(up.strip())
        ):
            up = persona
    pn = (persona or "").strip()
    if pn and not _is_literal_user_prompt_placeholder(pn) and "{user_prompt}" in (up or ""):
        up = (up or "").replace("{user_prompt}", pn)
    eval_vars = merge_template_vars(
        ctx,
        goals=goals,
        history=history_text,
        user_prompt=up,
        assistant_prompt=ap,
        available_tools=available_tools_description,
    )
    return ap, up, eval_vars


def summarize_context_fields(test_cases: List[dict]) -> dict:
    """
    Ключи context по всем кейсам и до 3 примеров значения на ключ (уникальные строки).
    """
    key_examples: Dict[str, List[str]] = {}
    for case in test_cases:
        ctx = case.get("context")
        if not isinstance(ctx, dict):
            continue
        for k, v in ctx.items():
            if not isinstance(k, str):
                continue
            if k not in key_examples:
                key_examples[k] = []
            if len(key_examples[k]) >= 3:
                continue
            if isinstance(v, (dict, list)):
                try:
                    s = json.dumps(v, ensure_ascii=False)
                except Exception:
                    s = str(v)
            else:
                s = "" if v is None else str(v)
            if s not in key_examples[k]:
                key_examples[k].append(s)
    return dict(sorted(key_examples.items()))
