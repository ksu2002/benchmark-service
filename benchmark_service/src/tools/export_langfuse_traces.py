#!/usr/bin/env python3
"""
Выгрузка трассировок из Langfuse с фильтрами и преобразованием в формат кейсов LLM-судьи.

Примеры:

  # Через переменные окружения (см. .env.template)
  export LANGFUSE_HOST=https://[NDA_LANGFUSE_HOST]
  export LANGFUSE_PUBLIC_KEY=pk-lf-...
  export LANGFUSE_SECRET_KEY=sk-lf-...

  python src/export_langfuse_traces.py \\
    --from 2026-06-01 --to 2026-06-22 \\
    --tag benchmark --tag llm-judge \\
    --format judge \\
    --require-history \\
    -o langfuse_judge_cases.jsonl

  python src/export_langfuse_traces.py \\
    --session-id my-session-123 \\
    --metadata-key dialog_id --metadata-value abc-123 \\
    --format raw \\
    -o traces_raw.jsonl

Формат ``judge`` — JSONL, совместимый с загрузкой в бенчмарк / калибровку LLM-судьи
(поля dialog_id, goals, context, history + блок langfuse с метаданными трассировки).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Sequence
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv

DEFAULT_PAGE_SIZE = 50
DEFAULT_REQUEST_TIMEOUT_SEC = 60


def _iso_timestamp(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def parse_datetime_arg(value: str) -> datetime:
    """Парсит YYYY-MM-DD или ISO-8601."""
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(value.replace("Z", ""), fmt.replace("Z", ""))
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Неверный формат даты: {value!r}. Используйте YYYY-MM-DD или ISO-8601."
    )


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


class LangfuseClient:
    """Минимальный клиент Public API Langfuse (без зависимости langfuse-sdk)."""

    def __init__(
        self,
        host: str,
        public_key: str,
        secret_key: str,
        timeout_sec: int = DEFAULT_REQUEST_TIMEOUT_SEC,
    ) -> None:
        self.base_url = host.rstrip("/") + "/"
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Basic {token}",
                "Content-Type": "application/json",
            }
        )
        self.timeout_sec = timeout_sec

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = urljoin(self.base_url, path.lstrip("/"))
        resp = self.session.get(url, params=params or {}, timeout=self.timeout_sec)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Langfuse API {resp.status_code} для {url}: {resp.text[:500]}"
            )
        return resp.json()

    def list_traces(
        self,
        *,
        page: int = 1,
        limit: int = DEFAULT_PAGE_SIZE,
        user_id: Optional[str] = None,
        name: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        trace_id: Optional[str] = None,
        filter_json: Optional[str] = None,
        fields: Optional[str] = None,
    ) -> dict:
        params: Dict[str, Any] = {"page": page, "limit": limit}
        if user_id:
            params["userId"] = user_id
        if name:
            params["name"] = name
        if session_id:
            params["sessionId"] = session_id
        if tags:
            params["tags"] = list(tags)
        if from_timestamp:
            params["fromTimestamp"] = _iso_timestamp(from_timestamp)
        if to_timestamp:
            params["toTimestamp"] = _iso_timestamp(to_timestamp)
        if trace_id:
            params["traceId"] = trace_id
        if filter_json:
            params["filter"] = filter_json
        if fields:
            params["fields"] = fields
        return self._get("api/public/traces", params)

    def get_trace(self, trace_id: str) -> dict:
        return self._get(f"api/public/traces/{trace_id}")

    def list_observations(
        self,
        *,
        trace_id: Optional[str] = None,
        page: int = 1,
        limit: int = 100,
        observation_type: Optional[str] = None,
        name: Optional[str] = None,
        fields: Optional[str] = None,
    ) -> dict:
        params: Dict[str, Any] = {
            "page": page,
            "limit": limit,
        }
        if trace_id:
            params["traceId"] = trace_id
        if observation_type:
            params["type"] = observation_type
        if name:
            params["name"] = name
        if fields:
            params["fields"] = fields
        return self._get("api/public/observations", params)

    def iter_observation_trace_ids(
        self,
        *,
        observation_type: str,
        observation_name: Optional[str] = None,
        sleep_sec: float = 0.0,
    ) -> Iterator[str]:
        """Уникальные traceId, у которых есть observation заданного type."""
        page = 1
        seen: set[str] = set()
        while True:
            payload = self.list_observations(
                page=page,
                limit=100,
                observation_type=observation_type,
                name=observation_name,
            )
            items = payload.get("data") or []
            if not items:
                break
            for item in items:
                tid = item.get("traceId")
                if tid and tid not in seen:
                    seen.add(tid)
                    yield tid
            meta = payload.get("meta") or {}
            total_pages = meta.get("totalPages")
            if total_pages is not None and page >= total_pages:
                break
            if len(items) < 100:
                break
            page += 1
            if sleep_sec > 0:
                time.sleep(sleep_sec)

    def iter_traces(
        self,
        *,
        limit_total: Optional[int] = None,
        sleep_sec: float = 0.0,
        **list_kwargs: Any,
    ) -> Iterator[dict]:
        page = 1
        fetched = 0
        while True:
            payload = self.list_traces(page=page, **list_kwargs)
            items = payload.get("data") or []
            if not items:
                break
            for item in items:
                yield item
                fetched += 1
                if limit_total is not None and fetched >= limit_total:
                    return
            meta = payload.get("meta") or {}
            total_pages = meta.get("totalPages")
            if total_pages is not None and page >= total_pages:
                break
            if len(items) < list_kwargs.get("limit", DEFAULT_PAGE_SIZE):
                break
            page += 1
            if sleep_sec > 0:
                time.sleep(sleep_sec)

    def iter_observations_for_trace(
        self,
        trace_id: str,
        *,
        observation_type: Optional[str] = None,
        name: Optional[str] = None,
        fields: Optional[str] = None,
        sleep_sec: float = 0.0,
    ) -> List[dict]:
        page = 1
        out: List[dict] = []
        while True:
            payload = self.list_observations(
                trace_id=trace_id,
                page=page,
                limit=100,
                observation_type=observation_type,
                name=name,
                fields=fields,
            )
            items = payload.get("data") or []
            out.extend(items)
            meta = payload.get("meta") or {}
            total_pages = meta.get("totalPages")
            if not items or (total_pages is not None and page >= total_pages):
                break
            page += 1
            if sleep_sec > 0:
                time.sleep(sleep_sec)
        return out


def build_advanced_filter(
    metadata_filters: Sequence[tuple[str, str]],
    *,
    name_contains: Optional[str] = None,
    tag: Optional[str] = None,
) -> Optional[str]:
    """Собирает JSON-filter для Langfuse Public API."""
    conditions: List[dict] = []
    for key, value in metadata_filters:
        conditions.append(
            {
                "type": "stringObject",
                "column": "metadata",
                "key": key,
                "operator": "=",
                "value": value,
            }
        )
    if name_contains:
        conditions.append(
            {
                "type": "string",
                "column": "name",
                "operator": "contains",
                "value": name_contains,
            }
        )
    if tag:
        conditions.append(
            {
                "type": "arrayOptions",
                "column": "tags",
                "operator": "contains",
                "value": tag,
            }
        )
    if not conditions:
        return None
    return json.dumps(conditions, ensure_ascii=False)


def _normalize_goals(raw: Any) -> Any:
    if raw is None:
        return ""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return [text] if text else ""
    return str(raw)


def _message_from_observation(obs: dict) -> Optional[dict]:
    """Пытается извлечь role/content из observation input/output."""
    name = (obs.get("name") or "").lower()
    obs_type = (obs.get("type") or "").upper()

    role = None
    if "user" in name:
        role = "user"
    elif "assistant" in name or obs_type == "GENERATION":
        role = "assistant"
    elif "tool" in name:
        role = "tool"

    content = None
    for source in (obs.get("output"), obs.get("input")):
        if source is None:
            continue
        if isinstance(source, str):
            content = source
            break
        if isinstance(source, dict):
            for key in ("content", "text", "message", "response"):
                if key in source and source[key]:
                    content = str(source[key])
                    break
            if content is None:
                content = json.dumps(source, ensure_ascii=False)
            break
        if isinstance(source, list):
            content = json.dumps(source, ensure_ascii=False)
            break

    if not content:
        return None
    if role is None:
        role = "assistant" if obs_type == "GENERATION" else "user"
    msg: dict = {"role": role, "content": content}
    if obs.get("output") is not None:
        msg["full_response"] = obs.get("output")
    return msg


def observations_to_history(observations: Sequence[dict]) -> List[dict]:
    history: List[dict] = []
    for obs in sorted(observations, key=lambda o: o.get("startTime") or ""):
        msg = _message_from_observation(obs)
        if msg:
            history.append(msg)
    return history


def _format_tool_calls_text(tool_calls: Sequence[dict]) -> str:
    parts: List[str] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        name = tc.get("name") or "?"
        args = tc.get("args")
        if isinstance(args, dict) and args:
            parts.append(f"→ {name}({json.dumps(args, ensure_ascii=False)})")
        else:
            parts.append(f"→ {name}()")
    return " ".join(parts)


def _format_tool_body(content: Any) -> str:
    if content is None or content == "":
        return "(пусто)"
    if isinstance(content, list):
        if not content:
            return "(нет данных)"
        if all(isinstance(x, (dict, str, int, float, bool)) or x is None for x in content):
            try:
                return json.dumps(content, ensure_ascii=False)
            except TypeError:
                pass
        text = str(content)
        return text if len(text) <= 400 else f"{len(content)} элементов"
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, str):
        text = content.strip()
        if text == "[]":
            return "(нет данных)"
        if text.startswith("{") or text.startswith("["):
            try:
                return json.dumps(json.loads(text), ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        return text
    return str(content)


def _langchain_message_to_turn(msg: dict) -> Optional[dict]:
    """Сообщение LangChain/LangGraph → ``{role, content}`` для LLM-судьи."""
    turns = _langchain_messages_to_turns(msg)
    return turns[0] if turns else None


def _langchain_messages_to_turns(msg: dict) -> List[dict]:
    msg_type = (msg.get("type") or "").lower()
    if msg_type == "human":
        text = str(msg.get("content") or "").strip()
        return [{"role": "user", "content": text}] if text else []

    if msg_type == "ai":
        content = str(msg.get("content") or "").strip()
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            calls_line = _format_tool_calls_text(tool_calls)
            if content:
                text = f"{content}\n{calls_line}"
            else:
                text = calls_line
            return [{"role": "assistant", "content": text, "tool_calls": tool_calls}]
        if content:
            return [{"role": "assistant", "content": content}]
        return []

    if msg_type == "tool":
        name = str(msg.get("name") or "tool")
        body = _format_tool_body(msg.get("content"))
        return [{"role": "tool", "tool_name": name, "content": f"{name}: {body}"}]

    role = msg.get("role") or "assistant"
    content = msg.get("content")
    if content is None or content == "":
        return []
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    if not str(text).strip():
        return []
    return [{"role": role, "content": str(text).strip()}]


def _history_from_langgraph_io(value: Any) -> List[dict]:
    """Разбирает trace.input / trace.output LangGraph (поле messages)."""
    if not isinstance(value, dict):
        return []
    messages = value.get("messages")
    if not isinstance(messages, list):
        return []
    history: List[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for turn in _langchain_messages_to_turns(msg):
            history.append(turn)
    return history


def _history_from_trace_io(trace: dict) -> List[dict]:
    history: List[dict] = []
    inp = trace.get("input")
    out = trace.get("output")

    langgraph_in = _history_from_langgraph_io(inp)
    langgraph_out = _history_from_langgraph_io(out)
    if langgraph_in or langgraph_out:
        if langgraph_out:
            return langgraph_out
        return langgraph_in

    if isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict) and "role" in item and "content" in item:
                history.append(item)
    elif isinstance(inp, dict):
        text = inp.get("content") or inp.get("message") or json.dumps(inp, ensure_ascii=False)
        history.append({"role": "user", "content": str(text)})
    elif isinstance(inp, str) and inp.strip():
        history.append({"role": "user", "content": inp.strip()})

    if isinstance(out, list):
        for item in out:
            if isinstance(item, dict) and "role" in item and "content" in item:
                history.append(item)
    elif isinstance(out, dict):
        text = out.get("content") or out.get("message") or json.dumps(out, ensure_ascii=False)
        history.append({"role": "assistant", "content": str(text)})
    elif isinstance(out, str) and out.strip():
        history.append({"role": "assistant", "content": out.strip()})
    return history


def trace_to_judge_case(trace: dict, observations: Optional[Sequence[dict]] = None) -> dict:
    """
    Преобразует trace Langfuse в кейс для LLM-судьи (совместим с benchmark_runner).
    """
    metadata = _as_dict(trace.get("metadata"))

    embedded = metadata.get("benchmark_case") or metadata.get("case")
    if isinstance(embedded, dict):
        case = dict(embedded)
        case.setdefault("langfuse_trace_id", trace.get("id"))
        return case

    goals = metadata.get("goals")
    if goals is None and trace.get("name"):
        goals = trace["name"]

    context = _as_dict(metadata.get("context"))
    for key in ("eval", "user_prompt_test", "address", "category", "scenario_id"):
        if key in metadata and key not in context:
            context[key] = metadata[key]

    history = metadata.get("history")
    if not history:
        history = _history_from_trace_io(trace)
    if not history and observations:
        history = observations_to_history(observations)

    dialog_id = (
        metadata.get("dialog_id")
        or trace.get("sessionId")
        or trace.get("id")
    )

    case: Dict[str, Any] = {
        "dialog_id": dialog_id,
        "goals": _normalize_goals(goals),
        "context": context,
        "history": _as_list(history),
        "langfuse": {
            "trace_id": trace.get("id"),
            "session_id": trace.get("sessionId"),
            "name": trace.get("name"),
            "user_id": trace.get("userId"),
            "tags": trace.get("tags") or [],
            "timestamp": trace.get("timestamp"),
            "metadata": metadata,
        },
    }

    if trace.get("input") is not None:
        case["langfuse"]["input"] = trace.get("input")
    if trace.get("output") is not None:
        case["langfuse"]["output"] = trace.get("output")

    for key in ("human_label", "human_labels", "human_note", "human_notes", "llm_label", "llm_reason"):
        if key in metadata:
            case[key] = metadata[key]

    return case


def trace_matches_post_filters(
    trace: dict,
    *,
    tags_all: Sequence[str],
    tags_any: Sequence[str],
    name_regex: Optional[str],
    metadata_contains: Sequence[tuple[str, str]],
) -> bool:
    import re

    trace_tags = set(trace.get("tags") or [])
    if tags_all and not set(tags_all).issubset(trace_tags):
        return False
    if tags_any and not (trace_tags & set(tags_any)):
        return False

    if name_regex:
        name = trace.get("name") or ""
        if not re.search(name_regex, name):
            return False

    metadata = _as_dict(trace.get("metadata"))
    metadata_text = json.dumps(metadata, ensure_ascii=False)
    for key, needle in metadata_contains:
        if key:
            val = metadata.get(key)
            hay = json.dumps(val, ensure_ascii=False) if val is not None else metadata_text
        else:
            hay = metadata_text
        if needle not in hay:
            return False
    return True


def _parse_trace_timestamp(trace: dict) -> Optional[datetime]:
    raw = trace.get("timestamp")
    if not raw:
        return None
    text = str(raw).strip().replace("Z", "+00:00")
    try:
        ts = datetime.fromisoformat(text)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def trace_matches_api_filters(trace: dict, filters: LangfuseExportFilters) -> bool:
    if filters.name and (trace.get("name") or "") != filters.name:
        return False
    if filters.session_id and (trace.get("sessionId") or "") != filters.session_id:
        return False
    if filters.user_id and (trace.get("userId") or "") != filters.user_id:
        return False
    ts = _parse_trace_timestamp(trace)
    if filters.from_timestamp and ts and ts < filters.from_timestamp:
        return False
    if filters.to_timestamp and ts and ts > filters.to_timestamp:
        return False
    if filters.from_timestamp and ts is None:
        return False
    return True


def _try_append_trace_row(
    rows: List[dict],
    client: LangfuseClient,
    trace: dict,
    filters: LangfuseExportFilters,
) -> bool:
    if not trace_matches_api_filters(trace, filters):
        return False
    if not trace_matches_post_filters(
        trace,
        tags_all=filters.tag_all,
        tags_any=filters.tag_any or filters.tags,
        name_regex=filters.name_regex,
        metadata_contains=filters.metadata_contains,
    ):
        return False
    row = _trace_row_from_api(
        client,
        trace,
        fmt=filters.fmt,
        include_observations=filters.include_observations,
        observation_type=filters.observation_type,
        observation_name=filters.observation_name,
        require_history=filters.require_history,
        sleep_sec=filters.sleep_sec,
    )
    if row is None:
        return False
    rows.append(row)
    return True


def _parse_metadata_contains(items: Sequence[str]) -> List[tuple[str, str]]:
    out: List[tuple[str, str]] = []
    for item in items:
        if ":" in item:
            key, val = item.split(":", 1)
            out.append((key.strip(), val))
        else:
            out.append(("", item))
    return out


@dataclass
class LangfuseExportFilters:
    host: str
    public_key: str
    secret_key: str
    fmt: str = "judge"
    from_timestamp: Optional[datetime] = None
    to_timestamp: Optional[datetime] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    name: Optional[str] = None
    name_contains: Optional[str] = None
    name_regex: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    tag_all: List[str] = field(default_factory=list)
    tag_any: List[str] = field(default_factory=list)
    metadata_filters: List[tuple[str, str]] = field(default_factory=list)
    metadata_contains: List[tuple[str, str]] = field(default_factory=list)
    filter_json: Optional[str] = None
    trace_ids: List[str] = field(default_factory=list)
    filter_observation_type: Optional[str] = None
    limit: Optional[int] = None
    page_size: int = DEFAULT_PAGE_SIZE
    fields: str = "core,io,scores,observations"
    include_observations: bool = False
    observation_type: Optional[str] = None
    observation_name: Optional[str] = None
    require_history: bool = False
    dedupe_by_thread_id: bool = False
    filter_tool_name: Optional[str] = None
    sleep_sec: float = 0.05
    timeout_sec: int = DEFAULT_REQUEST_TIMEOUT_SEC


def _case_history_len(case: dict) -> int:
    return len(case.get("history") or [])


def conversation_key_from_trace(trace: dict) -> str:
    """
    Ключ одного разговора: id первой реплики пользователя в messages.
    thread_id в LangGraph — имя воркера, не id диалога.
    """
    for source in (trace.get("output"), trace.get("input")):
        if not isinstance(source, dict):
            continue
        messages = source.get("messages")
        if not isinstance(messages, list):
            continue
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if (msg.get("type") or "").lower() != "human":
                continue
            mid = msg.get("id")
            content = str(msg.get("content") or "").strip()
            if mid:
                return str(mid)
            if content:
                return content[:200]
            break
    tid = trace.get("id")
    return str(tid) if tid else ""


def conversation_key_from_case(case: dict) -> str:
    lf = case.get("langfuse") if isinstance(case.get("langfuse"), dict) else {}
    trace_like = {
        "input": lf.get("input"),
        "output": lf.get("output"),
        "id": lf.get("trace_id"),
    }
    key = conversation_key_from_trace(trace_like)
    return key or str(case.get("dialog_id") or "")


def dedupe_cases_by_conversation(cases: Sequence[dict]) -> List[dict]:
    """
    Один кейс на разговор: шаги LangGraph с одной первой репликой пользователя
    объединяются; остаётся trace с самой длинной history.
    """
    best: Dict[str, tuple[int, str, dict]] = {}
    no_key: List[dict] = []
    for case in cases:
        conv_key = conversation_key_from_case(case)
        lf = case.get("langfuse") if isinstance(case.get("langfuse"), dict) else {}
        hist_len = _case_history_len(case)
        ts = str(lf.get("timestamp") or "")
        if not conv_key:
            no_key.append(case)
            continue
        prev = best.get(conv_key)
        if prev is None or hist_len > prev[0] or (hist_len == prev[0] and ts >= prev[1]):
            row = dict(case)
            trace_id = str(lf.get("trace_id") or "")
            row["dialog_id"] = trace_id or conv_key
            lf_copy = dict(lf)
            lf_copy["conversation_key"] = conv_key
            row["langfuse"] = lf_copy
            best[conv_key] = (hist_len, ts, row)
    out = [pair[2] for pair in sorted(best.values(), key=lambda x: x[1], reverse=True)]
    out.extend(no_key)
    return out


def dedupe_cases_by_thread_id(cases: Sequence[dict]) -> List[dict]:
    """Псевдоним: используйте ``dedupe_cases_by_conversation``."""
    return dedupe_cases_by_conversation(cases)


def _tool_names_from_langgraph_messages(messages: Sequence[dict]) -> set[str]:
    names: set[str] = set()
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if (msg.get("type") or "").lower() == "tool" and msg.get("name"):
            names.add(str(msg["name"]))
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict) and tc.get("name"):
                names.add(str(tc["name"]))
    return names


def case_has_tool_name(case: dict, tool_name: str) -> bool:
    needle = tool_name.strip().lower()
    if not needle:
        return True
    lf = case.get("langfuse") if isinstance(case.get("langfuse"), dict) else {}
    for source in (lf.get("output"), lf.get("input")):
        if not isinstance(source, dict):
            continue
        messages = source.get("messages")
        if isinstance(messages, list):
            found = {n.lower() for n in _tool_names_from_langgraph_messages(messages)}
            if needle in found:
                return True
    for turn in case.get("history") or []:
        if not isinstance(turn, dict):
            continue
        tn = turn.get("tool_name")
        if isinstance(tn, str) and tn.lower() == needle:
            return True
        for tc in turn.get("tool_calls") or []:
            if isinstance(tc, dict) and str(tc.get("name") or "").lower() == needle:
                return True
        if turn.get("role") == "tool":
            content = str(turn.get("content") or "")
            if content.lower().startswith(f"{needle}:"):
                return True
    return False


def apply_case_defaults(
    cases: Sequence[dict],
    *,
    default_eval: str = "",
    default_goals: str = "",
) -> List[dict]:
    out: List[dict] = []
    goals_override = default_goals.strip()
    eval_override = default_eval.strip()
    for case in cases:
        row = dict(case)
        if goals_override:
            row["goals"] = [goals_override]
        ctx = dict(row.get("context") or {}) if isinstance(row.get("context"), dict) else {}
        if eval_override and not str(ctx.get("eval") or "").strip():
            ctx["eval"] = eval_override
            row["context"] = ctx
        out.append(row)
    return out


def _trace_row_from_api(
    client: LangfuseClient,
    trace: dict,
    *,
    fmt: str,
    include_observations: bool,
    observation_type: Optional[str],
    observation_name: Optional[str],
    require_history: bool,
    sleep_sec: float,
) -> Optional[dict]:
    observations: List[dict] = []
    if include_observations or fmt == "judge":
        observations = client.iter_observations_for_trace(
            trace["id"],
            observation_type=observation_type,
            name=observation_name,
            fields="core,basic,io",
            sleep_sec=sleep_sec,
        )
    if fmt == "raw":
        row = dict(trace)
        if include_observations:
            row["observations"] = observations
        return row
    if fmt == "judge":
        case = trace_to_judge_case(trace, observations)
        if require_history and not case.get("history"):
            return None
        return case
    raise ValueError(f"Неизвестный формат: {fmt}")


def fetch_langfuse_traces(filters: LangfuseExportFilters) -> List[dict]:
    """Загружает трассировки из Langfuse и возвращает список кейсов/трасс."""
    filter_json = filters.filter_json or build_advanced_filter(
        filters.metadata_filters,
        name_contains=filters.name_contains,
        tag=(filters.tags[0] if len(filters.tags) == 1 and not filters.tag_all else None),
    )
    client = LangfuseClient(
        host=filters.host,
        public_key=filters.public_key,
        secret_key=filters.secret_key,
        timeout_sec=filters.timeout_sec,
    )
    rows: List[dict] = []

    def _at_limit() -> bool:
        if filters.dedupe_by_thread_id:
            return False
        return filters.limit is not None and len(rows) >= filters.limit

    if filters.trace_ids:
        for tid in filters.trace_ids:
            if _at_limit():
                break
            trace = client.get_trace(tid)
            _try_append_trace_row(rows, client, trace, filters)
    elif filters.filter_observation_type:
        obs_type = filters.filter_observation_type.strip().upper()
        for tid in client.iter_observation_trace_ids(
            observation_type=obs_type,
            observation_name=filters.observation_name,
            sleep_sec=filters.sleep_sec,
        ):
            if _at_limit():
                break
            trace = client.get_trace(tid)
            _try_append_trace_row(rows, client, trace, filters)
    else:
        list_kwargs: Dict[str, Any] = {
            "limit": filters.page_size,
            "limit_total": None if filters.dedupe_by_thread_id else filters.limit,
            "user_id": filters.user_id,
            "name": filters.name,
            "session_id": filters.session_id,
            "tags": filters.tags or None,
            "from_timestamp": filters.from_timestamp,
            "to_timestamp": filters.to_timestamp,
            "fields": filters.fields,
            "filter_json": filter_json,
        }
        for trace in client.iter_traces(sleep_sec=filters.sleep_sec, **list_kwargs):
            if not trace_matches_post_filters(
                trace,
                tags_all=filters.tag_all,
                tags_any=filters.tag_any or filters.tags,
                name_regex=filters.name_regex,
                metadata_contains=filters.metadata_contains,
            ):
                continue
            row = _trace_row_from_api(
                client,
                trace,
                fmt=filters.fmt,
                include_observations=filters.include_observations,
                observation_type=filters.observation_type,
                observation_name=filters.observation_name,
                require_history=filters.require_history,
                sleep_sec=filters.sleep_sec,
            )
            if row is not None:
                rows.append(row)

    if filters.dedupe_by_thread_id and filters.fmt == "judge":
        rows = dedupe_cases_by_conversation(rows)
    if filters.filter_tool_name:
        rows = [r for r in rows if case_has_tool_name(r, filters.filter_tool_name)]
    if filters.limit is not None:
        rows = rows[: filters.limit]
    return rows


def fetch_langfuse_dialog_cases(
    *,
    host: str,
    public_key: str,
    secret_key: str,
    from_timestamp: Optional[datetime] = None,
    to_timestamp: Optional[datetime] = None,
    filter_tool_name: Optional[str] = None,
    timeout_sec: int = DEFAULT_REQUEST_TIMEOUT_SEC,
) -> List[dict]:
    """
    Выгрузка целых диалогов из Langfuse за период.
    Шаги одного разговора объединяются; dialog_id = trace_id в Langfuse.
    """
    return fetch_langfuse_traces(
        LangfuseExportFilters(
            host=host,
            public_key=public_key,
            secret_key=secret_key,
            fmt="judge",
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
            require_history=True,
            dedupe_by_thread_id=True,
            filter_tool_name=(filter_tool_name or "").strip() or None,
            timeout_sec=timeout_sec,
        )
    )


def export_traces(
    client: LangfuseClient,
    *,
    output_path: Optional[str],
    fmt: str,
    include_observations: bool,
    observation_type: Optional[str],
    observation_name: Optional[str],
    require_history: bool,
    sleep_sec: float,
    dedupe_by_thread_id: bool = False,
    **list_kwargs: Any,
) -> int:
    post_tags_all = list_kwargs.pop("post_tags_all", [])
    post_tags_any = list_kwargs.pop("post_tags_any", [])
    post_name_regex = list_kwargs.pop("post_name_regex", None)
    post_metadata_contains = list_kwargs.pop("post_metadata_contains", [])

    rows = _export_traces_rows(
        client,
        fmt=fmt,
        include_observations=include_observations,
        observation_type=observation_type,
        observation_name=observation_name,
        require_history=require_history,
        sleep_sec=sleep_sec,
        dedupe_by_thread_id=dedupe_by_thread_id,
        post_tags_all=post_tags_all,
        post_tags_any=post_tags_any,
        post_name_regex=post_name_regex,
        post_metadata_contains=post_metadata_contains,
        **list_kwargs,
    )

    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    if text:
        text += "\n"

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)

    return len(rows)


def _export_traces_rows(
    client: LangfuseClient,
    *,
    fmt: str,
    include_observations: bool,
    observation_type: Optional[str],
    observation_name: Optional[str],
    require_history: bool,
    sleep_sec: float,
    dedupe_by_thread_id: bool,
    post_tags_all: Sequence[str],
    post_tags_any: Sequence[str],
    post_name_regex: Optional[str],
    post_metadata_contains: Sequence[tuple[str, str]],
    **list_kwargs: Any,
) -> List[dict]:
    rows: List[dict] = []
    for trace in client.iter_traces(sleep_sec=sleep_sec, **list_kwargs):
        if not trace_matches_post_filters(
            trace,
            tags_all=post_tags_all,
            tags_any=post_tags_any,
            name_regex=post_name_regex,
            metadata_contains=post_metadata_contains,
        ):
            continue
        row = _trace_row_from_api(
            client,
            trace,
            fmt=fmt,
            include_observations=include_observations,
            observation_type=observation_type,
            observation_name=observation_name,
            require_history=require_history,
            sleep_sec=sleep_sec,
        )
        if row is not None:
            rows.append(row)
    if dedupe_by_thread_id and fmt == "judge":
        rows = dedupe_cases_by_conversation(rows)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Выгрузка трассировок Langfuse с фильтрами для LLM-судьи.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--host", default=os.getenv("LANGFUSE_HOST"), help="URL Langfuse")
    p.add_argument(
        "--public-key",
        default=os.getenv("LANGFUSE_PUBLIC_KEY"),
        help="Public key (pk-lf-...)",
    )
    p.add_argument(
        "--secret-key",
        default=os.getenv("LANGFUSE_SECRET_KEY"),
        help="Secret key (sk-lf-...)",
    )

    p.add_argument(
        "--format",
        choices=("judge", "raw"),
        default="judge",
        help="judge — кейсы для LLM-судьи; raw — трассировки как в API",
    )
    p.add_argument("-o", "--output", help="Файл JSONL (иначе stdout)")

    p.add_argument("--from", dest="from_ts", type=parse_datetime_arg, help="Начало периода (UTC)")
    p.add_argument("--to", dest="to_ts", type=parse_datetime_arg, help="Конец периода (UTC)")
    p.add_argument("--user-id", help="Фильтр userId")
    p.add_argument("--session-id", help="Фильтр sessionId")
    p.add_argument("--name", help="Точное имя trace (API filter)")
    p.add_argument("--name-contains", help="Подстрока в name (advanced filter)")
    p.add_argument("--name-regex", help="Regex для name (пост-фильтр)")
    p.add_argument("--trace-id", action="append", default=[], help="Конкретный trace id (можно несколько)")
    p.add_argument("--tag", action="append", default=[], dest="tags", help="Тег (API, можно несколько)")
    p.add_argument(
        "--tag-all",
        action="append",
        default=[],
        help="Все перечисленные теги должны быть у trace (пост-фильтр)",
    )
    p.add_argument(
        "--tag-any",
        action="append",
        default=[],
        help="Хотя бы один из тегов (пост-фильтр)",
    )
    p.add_argument(
        "--metadata-key",
        action="append",
        default=[],
        help="Ключ metadata для фильтра (=), пару задаёт --metadata-value",
    )
    p.add_argument(
        "--metadata-value",
        action="append",
        default=[],
        help="Значение metadata для фильтра (=)",
    )
    p.add_argument(
        "--metadata-contains",
        action="append",
        default=[],
        metavar="KEY:TEXT",
        help="Подстрока в metadata[key] или во всём metadata, если KEY пуст",
    )
    p.add_argument(
        "--filter-json",
        help="Готовый JSON advanced filter (перекрывает metadata-key/value)",
    )
    p.add_argument("--limit", type=int, help="Максимум трассировок")
    p.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="Размер страницы API")
    p.add_argument(
        "--fields",
        default="core,io,scores,observations",
        help="Поля trace в API (меньше — быстрее)",
    )
    p.add_argument(
        "--include-observations",
        action="store_true",
        help="Добавить observations (для raw; для judge подтягиваются автоматически)",
    )
    p.add_argument("--observation-type", help="GENERATION, SPAN, EVENT, … (observations внутри trace)")
    p.add_argument(
        "--filter-observation-type",
        help="Type observation для отбора trace: AGENT, CHAIN, GENERATION, SPAN, TOOL, EVENT",
    )
    p.add_argument("--observation-name", help="Имя observation")
    p.add_argument(
        "--require-history",
        action="store_true",
        help="Пропускать judge-кейсы без history",
    )
    p.add_argument(
        "--dedupe-by-thread-id",
        action="store_true",
        help="Оставить последний trace на каждый metadata.thread_id",
    )
    p.add_argument(
        "--sleep-sec",
        type=float,
        default=0.05,
        help="Пауза между страницами API",
    )
    p.add_argument(
        "--timeout-sec",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SEC,
        help="Таймаут HTTP-запроса",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    args = build_arg_parser().parse_args(argv)

    missing = [
        name
        for name, val in (
            ("--host / LANGFUSE_HOST", args.host),
            ("--public-key / LANGFUSE_PUBLIC_KEY", args.public_key),
            ("--secret-key / LANGFUSE_SECRET_KEY", args.secret_key),
        )
        if not val
    ]
    if missing:
        print("Не заданы параметры Langfuse:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 2

    metadata_filters: List[tuple[str, str]] = []
    if not args.filter_json:
        keys = args.metadata_key or []
        values = args.metadata_value or []
        if len(keys) != len(values):
            print(
                "Число --metadata-key должно совпадать с --metadata-value",
                file=sys.stderr,
            )
            return 2
        metadata_filters = list(zip(keys, values))

    filter_json = args.filter_json or build_advanced_filter(
        metadata_filters,
        name_contains=args.name_contains,
        tag=(args.tags[0] if len(args.tags) == 1 and not args.tag_all else None),
    )

    filters = LangfuseExportFilters(
        host=args.host,
        public_key=args.public_key,
        secret_key=args.secret_key,
        fmt=args.format,
        from_timestamp=args.from_ts,
        to_timestamp=args.to_ts,
        user_id=args.user_id,
        session_id=args.session_id,
        name=args.name,
        name_contains=args.name_contains,
        name_regex=args.name_regex,
        tags=list(args.tags or []),
        tag_all=list(args.tag_all or []),
        tag_any=list(args.tag_any or []),
        metadata_filters=metadata_filters,
        metadata_contains=_parse_metadata_contains(args.metadata_contains),
        filter_json=filter_json,
        trace_ids=list(args.trace_id or []),
        filter_observation_type=(args.filter_observation_type or "").strip().upper() or None,
        limit=args.limit,
        page_size=args.page_size,
        fields=args.fields,
        include_observations=args.include_observations,
        observation_type=args.observation_type,
        observation_name=args.observation_name,
        require_history=args.require_history,
        dedupe_by_thread_id=args.dedupe_by_thread_id,
        sleep_sec=args.sleep_sec,
        timeout_sec=args.timeout_sec,
    )
    rows = fetch_langfuse_traces(filters)
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    if text:
        text += "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    dest = args.output or "stdout"
    print(f"Экспортировано записей: {len(rows)} → {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
