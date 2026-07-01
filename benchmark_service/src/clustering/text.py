"""Общие текстовые и полевые утилиты для кластеризации диалогов."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import MutableMapping
from io import StringIO
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)

CLUSTERING_ALGORITHM_LABEL = "Кластеризация"
DEFAULT_EMBEDDING_API = (
    "https://[NDA_EMBEDDING_HOST]/embedding"
)
EMBEDDING_BATCH_SIZE = 10
EMPTY_FIELD_PLACEHOLDER_TMPL = "[пустой {field}]"
DEFAULT_DIALOG_EMBED_CHARS = 8000
FIELD_EMBED_CHARS = 1000
HISTORY_CLUSTER_FIELDS = frozenset({"dialog", "assistant", "user"})
CLUSTER_FIELD_LABELS: Dict[str, str] = {
    "dialog": "Диалог",
    "assistant": "Фразы ассистента",
    "user": "Фразы пользователя",
}
DEDUP_FIELD_LABELS: Dict[str, str] = dict(CLUSTER_FIELD_LABELS)
CUSTOM_RECORD_FIELD_SENTINEL = "__custom__"
RECORD_FIELD_SELECT_LABELS: Dict[str, str] = {
    **CLUSTER_FIELD_LABELS,
    CUSTOM_RECORD_FIELD_SENTINEL: "Своё поле (ключ в JSONL)",
}
_RESERVED_RECORD_FIELDS = frozenset({"history", "cluster_id", "cluster_label", "topic_keywords"})
ProgressCallback = Callable[[str, float], None]

_RU_STOP_WORDS = frozenset({"и","в","во","на","с","со","по","для","не","нет","да","что","это","как","из","у","к","ко","о","об","а","но","или","же","ли","бы","то","все","всё","всего","уже","ещё","еще","если","так","там","тут","при","про","над","под","от","до","без","через","я","мы","вы","он","она","они","оно","мне","меня","мой","моя","мои","вас","вам","ваш","ваша","ваши","нам","нас","наш","наша","наши","the","a","an","is","are","was","were","be","to","of","in","on","at","for","with","and","or","not","this","that","it","you","your","i","we","they","none","null","nan","true","false","unknown","undefined","assistant","user","bot","system","здравствуйте","подскажите","пожалуйста","можно","можем"})
_JUNK_TFIDF_TERMS = frozenset({"none","null","nan","true","false","unknown","undefined","assistant","user","bot","system","empty","пусто"})
_VECTOR_TOKEN_PATTERN = r"(?u)\b[^\W\d_]\w+\b"
_CTFIDF_NGRAM_RANGE = (1, 1)
_CTFIDF_MAX_TERM_LEN = 24
_NULLISH_TOKEN_RE = re.compile(r"\b(?:none|null|nan|undefined|true|false)\b", re.IGNORECASE)


def _is_meaningful_tfidf_term(term: str) -> bool:
    text = (term or "").strip()
    if len(text) < 2 or len(text) > _CTFIDF_MAX_TERM_LEN or " " in text:
        return False
    if text.casefold() in _JUNK_TFIDF_TERMS or text.casefold() in _RU_STOP_WORDS or text.isdigit():
        return False
    return bool(re.search(r"[^\W\d_]", text, flags=re.UNICODE))


def _topic_label_is_meaningful(label: str) -> bool:
    raw = (label or "").strip()
    if not raw or re.fullmatch(r"Тема\s*-?\d+", raw, flags=re.IGNORECASE):
        return False
    return any(_is_meaningful_tfidf_term(p.strip()) for p in raw.split(","))


def _filter_tfidf_term_pairs(words: Sequence[Tuple[str, float]], *, top_n: int = 10) -> List[Tuple[str, float]]:
    meaningful = [(str(word), float(score)) for word, score in words if _is_meaningful_tfidf_term(str(word))]
    filtered: List[Tuple[str, float]] = []
    for term, score in sorted(meaningful, key=lambda item: item[1], reverse=True):
        if any(term.casefold() in kept.casefold() for kept, _ in filtered):
            continue
        filtered.append((term, score))
        if len(filtered) >= top_n:
            break
    return filtered


def embedding_api_url() -> str:
    return (os.getenv("EMBEDDING_API_URL") or DEFAULT_EMBEDDING_API).strip()


def field_key(raw: object) -> str:
    return "" if raw is None else str(raw).strip()


def sanitize_cluster_text(text: str) -> str:
    if not text:
        return ""
    cleaned = _NULLISH_TOKEN_RE.sub(" ", text)
    cleaned = re.sub(r"([.!?;,])([^\s])", r"\1 \2", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _ctfidf_stop_words() -> List[str]:
    return sorted(_RU_STOP_WORDS | _JUNK_TFIDF_TERMS)


def _clean_metadata_value(raw: object) -> str:
    key = field_key(raw)
    if not key or key.casefold() in _JUNK_TFIDF_TERMS:
        return ""
    return key


def cluster_field_label(cluster_field: str) -> str:
    return record_field_label(cluster_field)


def is_history_cluster_field(cluster_field: str) -> bool:
    return cluster_field in HISTORY_CLUSTER_FIELDS


def is_valid_custom_record_field(name: str) -> bool:
    field = (name or "").strip()
    if not field or field in HISTORY_CLUSTER_FIELDS or field in _RESERVED_RECORD_FIELDS:
        return False
    return bool(re.fullmatch(r"[\w.-]+", field, flags=re.UNICODE))


def resolve_record_field(select_value: str, custom_value: str) -> str:
    return (custom_value or "").strip() if select_value == CUSTOM_RECORD_FIELD_SENTINEL else (select_value or "").strip()


def validate_record_field(field: str) -> str:
    field = (field or "").strip()
    if not field:
        raise ValueError("Укажите поле записи JSONL.")
    if field in HISTORY_CLUSTER_FIELDS:
        return field
    if not is_valid_custom_record_field(field):
        raise ValueError(f"Недопустимое имя поля {field!r}. Используйте буквы, цифры, «_», «-», «.»")
    return field


def record_field_label(field: str) -> str:
    return CLUSTER_FIELD_LABELS.get(field, field)


def turn_scope_label(turn_index: int) -> str:
    if turn_index <= 0:
        return "все фразы"
    return f"{turn_index}-я фраза"


def field_scope_label(field: str, turn_index: int = 0) -> str:
    base = record_field_label(field)
    return f"{base} ({turn_scope_label(turn_index)})" if is_history_cluster_field(field) and turn_index > 0 else base


def _turns_from_record(rec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    history = rec.get("history") or []
    return [t for t in history if isinstance(t, dict)] if isinstance(history, list) else []


def _turn_text(turn: Mapping[str, Any]) -> str:
    content = turn.get("content")
    if isinstance(content, str):
        return content.strip()
    return str(content).strip() if content is not None else ""


def extract_dedup_text(record: Mapping[str, Any], dedup_field: str, *, turn_index: int = 0) -> str:
    if dedup_field == "goals":
        goals = record.get("goals")
        if isinstance(goals, list):
            parts = [str(g).strip() for g in goals if g is not None and str(g).strip()]
            if parts:
                return "; ".join(parts)
        return field_key(goals)
    return extract_cluster_text(record, dedup_field, turn_index=turn_index)


def extract_cluster_text(record: Mapping[str, Any], cluster_field: str, *, turn_index: int = 0) -> str:
    if not is_history_cluster_field(cluster_field):
        return sanitize_cluster_text(field_key(record.get(cluster_field)))
    role_filter: Optional[frozenset[str]] = None
    if cluster_field == "assistant":
        role_filter = frozenset({"assistant"})
    elif cluster_field == "user":
        role_filter = frozenset({"user"})
    parts: List[str] = []
    for turn in _turns_from_record(record):
        role = (turn.get("role") or "").strip().lower()
        if role_filter is not None and role not in role_filter:
            continue
        text = _turn_text(turn)
        if text:
            parts.append(text)
    if turn_index > 0:
        idx = turn_index - 1
        return sanitize_cluster_text(parts[idx]) if idx < len(parts) else ""
    return sanitize_cluster_text(" ".join(parts))


def max_chars_for_cluster_field(cluster_field: str) -> int:
    return DEFAULT_DIALOG_EMBED_CHARS if is_history_cluster_field(cluster_field) else FIELD_EMBED_CHARS


def empty_placeholder_for_field(field_name: str) -> str:
    return EMPTY_FIELD_PLACEHOLDER_TMPL.format(field=field_name)


def text_for_embedding(key: str, *, empty_placeholder: str, max_len: int = FIELD_EMBED_CHARS) -> str:
    return empty_placeholder if not key else key[:max_len]


def text_for_cluster_model(raw: str, *, empty_placeholder: str, max_len: int = FIELD_EMBED_CHARS) -> str:
    cleaned = sanitize_cluster_text(raw) or (raw or "").strip()
    return text_for_embedding(cleaned, empty_placeholder=empty_placeholder, max_len=max_len)


def safe_json_write(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: safe_json_write(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json_write(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def load_jsonl_records(text: str) -> Tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []
    bad = 0
    for lineno, line in enumerate(text.splitlines(), start=1):
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith("#"):
            continue
        try:
            obj = json.loads(line_stripped)
        except json.JSONDecodeError as e:
            bad += 1
            logger.warning("Строка %s: JSON: %s", lineno, e)
            continue
        if not isinstance(obj, MutableMapping):
            bad += 1
            continue
        records.append(dict(obj))
    return records, bad


def _dialog_id_from_record(rec: Mapping[str, Any]) -> str:
    for key in ("dialog_id", "id", "idx"):
        val = rec.get(key)
        if val is not None and str(val).strip():
            return str(val)
    return "?"


def embeddings_by_dialog_id(
    records: Sequence[Mapping[str, Any]],
    embeddings: Sequence[Optional[List[float]]],
) -> Dict[str, Optional[List[float]]]:
    if len(records) != len(embeddings):
        return {}
    return {_dialog_id_from_record(rec): emb for rec, emb in zip(records, embeddings)}


def get_embeddings_batch(texts: List[str], *, on_progress: Optional[ProgressCallback] = None, max_chars: int = FIELD_EMBED_CHARS) -> List[Optional[List[float]]]:
    api_url = embedding_api_url()
    embeddings: List[Optional[List[float]]] = []
    total = max(1, (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE)
    for batch_idx, i in enumerate(range(0, len(texts), EMBEDDING_BATCH_SIZE)):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        batch_embeddings: List[Optional[List[float]]] = []
        for text in batch:
            try:
                clean = str(text).strip() if text else ""
                if not clean:
                    batch_embeddings.append(None)
                    continue
                resp = requests.post(api_url, json={"text": clean[:max_chars]}, headers={"accept": "application/json", "Content-Type": "application/json"}, timeout=60)
                resp.raise_for_status()
                emb = resp.json().get("embedding")
                batch_embeddings.append(emb if emb and len(emb) > 0 else None)
            except Exception as e:
                logger.warning("Ошибка эмбеддинга: %s", e)
                batch_embeddings.append(None)
        embeddings.extend(batch_embeddings)
        if on_progress:
            on_progress(f"Эмбеддинги: {min(i + len(batch), len(texts))}/{len(texts)}", (batch_idx + 1) / total)
    return embeddings
