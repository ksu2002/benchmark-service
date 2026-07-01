"""Семантическая кластеризация диалогов: эмбеддинги → UMAP → HDBSCAN."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import requests
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import Normalizer

from clustering import (
    CLUSTERING_ALGORITHM_LABEL,
    CUSTOM_RECORD_FIELD_SENTINEL,
    DEFAULT_DIALOG_EMBED_CHARS,
    DEFAULT_EMBEDDING_API,
    DEDUP_FIELD_LABELS,
    EMBEDDING_BATCH_SIZE,
    EMPTY_FIELD_PLACEHOLDER_TMPL,
    FIELD_EMBED_CHARS,
    HISTORY_CLUSTER_FIELDS,
    RECORD_FIELD_SELECT_LABELS,
    ProgressCallback,
    _clean_metadata_value,
    _ctfidf_stop_words,
    _dialog_id_from_record,
    _filter_tfidf_term_pairs,
    _is_meaningful_tfidf_term,
    _topic_label_is_meaningful,
    cluster_field_label,
    empty_placeholder_for_field,
    embedding_api_url,
    embeddings_by_dialog_id,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_key,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
    field_key,
    field_scope_label,
)

try:
    import umap
except ImportError:  # pragma: no cover
    umap = None  # type: ignore

logger = logging.getLogger(__name__)

CLUSTERING_ALGORITHM_LABEL = "Кластеризация"

DEFAULT_EMBEDDING_API = (
    "https://[NDA_EMBEDDING_HOST]/embedding"
)
EMBEDDING_BATCH_SIZE = 10
EMPTY_FIELD_PLACEHOLDER_TMPL = "[пустой {field}]"
DEFAULT_DIALOG_EMBED_CHARS = 8000
FIELD_EMBED_CHARS = 1000

# Поля кластеризации по тексту из history (один эмбеддинг на диалог).
HISTORY_CLUSTER_FIELDS = frozenset({"dialog", "assistant", "user"})

CLUSTER_FIELD_LABELS: Dict[str, str] = {
    "dialog": "Диалог",
    "assistant": "Фразы ассистента",
    "user": "Фразы пользователя",
}

# Те же стандартные поля, что и для кластеризации (+ своё поле через UI).
DEDUP_FIELD_LABELS: Dict[str, str] = dict(CLUSTER_FIELD_LABELS)

CUSTOM_RECORD_FIELD_SENTINEL = "__custom__"

RECORD_FIELD_SELECT_LABELS: Dict[str, str] = {
    **CLUSTER_FIELD_LABELS,
    CUSTOM_RECORD_FIELD_SENTINEL: "Своё поле (ключ в JSONL)",
}

_RESERVED_RECORD_FIELDS = frozenset(
    {"history", "cluster_id", "cluster_label", "topic_keywords"}
)

ProgressCallback = Callable[[str, float], None]


@dataclass
class UmapSettings:
    enabled: bool = True
    n_components: int = 0  # 0 — авто
    n_neighbors: int = 0  # 0 — авто
    min_dist: float = 0.001
    spread: float = 0.5
    metric: str = "cosine"
    random_state: int = 42
    init: str = "auto"  # auto | spectral | random


@dataclass
class HdbscanSettings:
    min_cluster_size: int = 5
    min_samples: int = 1
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    alpha: float = 1.0
    auto_scale_min_cluster_size: bool = True


@dataclass
class BertopicSettings:
    min_topic_size: int = 5
    min_samples: int = 1
    hdbscan_metric: str = "euclidean"
    cluster_selection_method: str = "eom"
    alpha: float = 1.0
    auto_scale_min_topic_size: bool = True
    reduce_outliers: bool = True
    outlier_strategy: str = "embeddings"
    reduce_topics: bool = False
    nr_topics: str = "auto"
    top_n_words: int = 5
    calculate_probabilities: bool = False
    use_ctfidf: bool = True


_RU_STOP_WORDS = frozenset(
    {
        "и",
        "в",
        "во",
        "на",
        "с",
        "со",
        "по",
        "для",
        "не",
        "нет",
        "да",
        "что",
        "это",
        "как",
        "из",
        "у",
        "к",
        "ко",
        "о",
        "об",
        "а",
        "но",
        "или",
        "же",
        "ли",
        "бы",
        "то",
        "все",
        "всё",
        "всего",
        "уже",
        "ещё",
        "еще",
        "если",
        "так",
        "там",
        "тут",
        "при",
        "про",
        "над",
        "под",
        "от",
        "до",
        "без",
        "через",
        "я",
        "мы",
        "вы",
        "он",
        "она",
        "они",
        "оно",
        "мне",
        "меня",
        "мой",
        "моя",
        "мои",
        "вас",
        "вам",
        "ваш",
        "ваша",
        "ваши",
        "нам",
        "нас",
        "наш",
        "наша",
        "наши",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "with",
        "and",
        "or",
        "not",
        "this",
        "that",
        "it",
        "you",
        "your",
        "i",
        "we",
        "they",
        "none",
        "null",
        "nan",
        "true",
        "false",
        "unknown",
        "undefined",
        "assistant",
        "user",
        "bot",
        "system",
        "здравствуйте",
        "подскажите",
        "пожалуйста",
        "можно",
        "можем",
    }
)

_JUNK_TFIDF_TERMS = frozenset(
    {
        "none",
        "null",
        "nan",
        "true",
        "false",
        "unknown",
        "undefined",
        "assistant",
        "user",
        "bot",
        "system",
        "empty",
        "пусто",
    }
)

# Токен должен начинаться с буквы — чистые числа (ID, суммы) не считаются словами темы.
_VECTOR_TOKEN_PATTERN = r"(?u)\b[^\W\d_]\w+\b"
_CTFIDF_NGRAM_RANGE = (1, 1)
_CTFIDF_MAX_TERM_LEN = 24

_NULLISH_TOKEN_RE = re.compile(
    r"\b(?:none|null|nan|undefined|true|false)\b",
    re.IGNORECASE,
)


def _is_meaningful_tfidf_term(term: str) -> bool:
    text = (term or "").strip()
    if len(text) < 2:
        return False
    if len(text) > _CTFIDF_MAX_TERM_LEN:
        return False
    if " " in text:
        return False
    if text.casefold() in _JUNK_TFIDF_TERMS:
        return False
    if text.casefold() in _RU_STOP_WORDS:
        return False
    if text.isdigit():
        return False
    if not re.search(r"[^\W\d_]", text, flags=re.UNICODE):
        return False
    return True


def _topic_label_is_meaningful(label: str) -> bool:
    raw = (label or "").strip()
    if not raw:
        return False
    if re.fullmatch(r"Тема\s*-?\d+", raw, flags=re.IGNORECASE):
        return False
    parts = [part.strip() for part in raw.split(",")]
    meaningful_parts = [part for part in parts if part and _is_meaningful_tfidf_term(part)]
    return bool(meaningful_parts)


def _filter_tfidf_term_pairs(
    words: Sequence[Tuple[str, float]],
    *,
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    meaningful = [
        (str(word), float(score))
        for word, score in words
        if _is_meaningful_tfidf_term(str(word))
    ]
    filtered: List[Tuple[str, float]] = []
    for term, score in sorted(meaningful, key=lambda item: item[1], reverse=True):
        lowered = term.casefold()
        if any(lowered in kept.casefold() for kept, _ in filtered):
            continue
        filtered.append((term, score))
        if len(filtered) >= top_n:
            break
    return filtered


def _resolve_umap_n_neighbors(requested: int, n_ok: int) -> int:
    if requested > 0:
        return max(2, min(requested, n_ok - 1))
    return max(2, min(10, max(2, n_ok // 5), n_ok - 1))


def _resolve_umap_n_components(requested: int, emb_dim: int, n_ok: int) -> int:
    if requested > 0:
        return max(2, min(requested, emb_dim, n_ok - 1))
    return min(50, emb_dim, max(2, n_ok - 1))


def _resolve_umap_init(init: str, n_ok: int) -> str:
    if init in ("spectral", "random"):
        return init
    return "random" if n_ok < 50 else "spectral"


def _apply_hdbscan(
    emb_proc: np.ndarray,
    *,
    n_ok: int,
    hdbscan: HdbscanSettings,
) -> np.ndarray:
    mcs = hdbscan.min_cluster_size
    if hdbscan.auto_scale_min_cluster_size:
        mcs = min(mcs, max(3, n_ok // 10))
    mcs = max(2, min(mcs, n_ok))
    ms_eff = max(1, min(hdbscan.min_samples, n_ok))
    clusterer = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms_eff,
        metric=hdbscan.metric,
        cluster_selection_method=hdbscan.cluster_selection_method,
        alpha=hdbscan.alpha,
        prediction_data=True,
    )
    return clusterer.fit_predict(emb_proc)


@dataclass
class ClusterTextsResult:
    labels: List[int]
    viz_x: List[Optional[float]]
    viz_y: List[Optional[float]]
    embeddings: List[Optional[List[float]]] = field(default_factory=list)


def _compute_viz_coords_2d(
    embeddings_array: np.ndarray,
    umap_cfg: UmapSettings,
) -> np.ndarray:
    """2D-проекция UMAP для визуализации (отдельно от UMAP в пайплайне кластеризации)."""
    n_ok = len(embeddings_array)
    if n_ok == 1:
        return np.array([[0.0, 0.0]])
    if n_ok == 2:
        return np.array([[0.0, 0.0], [1.0, 0.0]])

    if umap is None:
        raise ValueError(
            "Для 2D-визуализации установите umap-learn: pip install umap-learn"
        )

    n_neigh = _resolve_umap_n_neighbors(umap_cfg.n_neighbors, n_ok)
    init_umap = _resolve_umap_init(umap_cfg.init, n_ok)
    reducer = umap.UMAP(
        n_components=2,
        metric=umap_cfg.metric,
        random_state=umap_cfg.random_state,
        n_neighbors=n_neigh,
        min_dist=umap_cfg.min_dist,
        spread=umap_cfg.spread,
        verbose=False,
        init=init_umap,
    )
    return reducer.fit_transform(embeddings_array)


def embedding_api_url() -> str:
    return (os.getenv("EMBEDDING_API_URL") or DEFAULT_EMBEDDING_API).strip()


def _ctfidf_stop_words() -> List[str]:
    return sorted(_RU_STOP_WORDS | _JUNK_TFIDF_TERMS)


def _clean_metadata_value(raw: object) -> str:
    key = field_key(raw)
    if not key:
        return ""
    if key.casefold() in _JUNK_TFIDF_TERMS:
        return ""
    return key


def cluster_field_label(cluster_field: str) -> str:
    return record_field_label(cluster_field)


def is_history_cluster_field(cluster_field: str) -> bool:
    return cluster_field in HISTORY_CLUSTER_FIELDS


def is_valid_custom_record_field(name: str) -> bool:
    field = (name or "").strip()
    if not field or field in HISTORY_CLUSTER_FIELDS:
        return False
    if field in _RESERVED_RECORD_FIELDS:
        return False
    return bool(re.fullmatch(r"[\w.-]+", field, flags=re.UNICODE))


def resolve_record_field(select_value: str, custom_value: str) -> str:
    if select_value == CUSTOM_RECORD_FIELD_SENTINEL:
        return (custom_value or "").strip()
    return (select_value or "").strip()


def validate_record_field(field: str) -> str:
    field = (field or "").strip()
    if not field:
        raise ValueError("Укажите поле записи JSONL.")
    if field in HISTORY_CLUSTER_FIELDS:
        return field
    if not is_valid_custom_record_field(field):
        raise ValueError(
            f"Недопустимое имя поля {field!r}. "
            "Используйте буквы, цифры, «_», «-», «.» (поле на верхнем уровне JSONL, не history)."
        )
    return field


def record_field_label(field: str) -> str:
    if field in CLUSTER_FIELD_LABELS:
        return CLUSTER_FIELD_LABELS[field]
    return field


def turn_scope_label(turn_index: int) -> str:
    """0 — все фразы; 1 — первая; 2 — вторая и т.д."""
    if turn_index <= 0:
        return "все фразы"
    return f"{turn_index}-я фраза"


def field_scope_label(field: str, turn_index: int = 0) -> str:
    base = record_field_label(field)
    if is_history_cluster_field(field) and turn_index > 0:
        return f"{base} ({turn_scope_label(turn_index)})"
    return base


def _turns_from_record(rec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    history = rec.get("history") or []
    if isinstance(history, list):
        return [t for t in history if isinstance(t, dict)]
    return []


def _turn_text(turn: Mapping[str, Any]) -> str:
    content = turn.get("content")
    if isinstance(content, str):
        return content.strip()
    if content is not None:
        return str(content).strip()
    return ""


def extract_dedup_text(
    record: Mapping[str, Any],
    dedup_field: str,
    *,
    turn_index: int = 0,
) -> str:
    """Текст записи для поиска дубликатов (history или scalar-поле)."""
    if dedup_field == "goals":
        goals = record.get("goals")
        if isinstance(goals, list):
            parts = [str(g).strip() for g in goals if g is not None and str(g).strip()]
            if parts:
                return "; ".join(parts)
        return field_key(goals)
    return extract_cluster_text(record, dedup_field, turn_index=turn_index)


def extract_cluster_text(
    record: Mapping[str, Any],
    cluster_field: str,
    *,
    turn_index: int = 0,
) -> str:
    """Текст для эмбеддинга: реплики из history (все или одна по номеру)."""
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
        if idx < len(parts):
            return sanitize_cluster_text(parts[idx])
        return ""

    return sanitize_cluster_text(" ".join(parts))


def max_chars_for_cluster_field(cluster_field: str) -> int:
    if is_history_cluster_field(cluster_field):
        return DEFAULT_DIALOG_EMBED_CHARS
    return FIELD_EMBED_CHARS


def empty_placeholder_for_field(field_name: str) -> str:
    return EMPTY_FIELD_PLACEHOLDER_TMPL.format(field=field_name)


def text_for_embedding(key: str, *, empty_placeholder: str, max_len: int = FIELD_EMBED_CHARS) -> str:
    return empty_placeholder if not key else key[:max_len]


def text_for_cluster_model(
    raw: str,
    *,
    empty_placeholder: str,
    max_len: int = FIELD_EMBED_CHARS,
) -> str:
    """Единый текст для эмбеддинга и c-TF-IDF (с санитизацией и placeholder)."""
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


def get_embeddings_batch(
    texts: List[str],
    *,
    on_progress: Optional[ProgressCallback] = None,
    max_chars: int = FIELD_EMBED_CHARS,
) -> List[Optional[List[float]]]:
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
                resp = requests.post(
                    api_url,
                    json={"text": clean[:max_chars]},
                    headers={"accept": "application/json", "Content-Type": "application/json"},
                    timeout=60,
                )
                resp.raise_for_status()
                emb = resp.json().get("embedding")
                batch_embeddings.append(emb if emb and len(emb) > 0 else None)
            except Exception as e:
                logger.warning("Ошибка эмбеддинга: %s", e)
                batch_embeddings.append(None)
            time.sleep(0.05)
        embeddings.extend(batch_embeddings)
        if on_progress:
            on_progress(
                f"Эмбеддинги: {min(i + len(batch), len(texts))}/{len(texts)}",
                (batch_idx + 1) / total,
            )
    return embeddings


def diagnose_embeddings(embeddings_array: np.ndarray, *, sample_size: int = 100) -> Dict[str, float]:
    if len(embeddings_array) <= 1:
        return {}
    sample = embeddings_array[: min(sample_size, len(embeddings_array))]
    distances = cosine_distances(sample)
    np.fill_diagonal(distances, np.nan)
    flat = distances[~np.isnan(distances)]
    return {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
    }


def cluster_unique_texts(
    unique_keys: List[str],
    *,
    cluster_field: str,
    use_umap: bool = True,
    min_cluster_size: int = 5,
    min_samples: int = 1,
    on_progress: Optional[ProgressCallback] = None,
) -> Dict[str, int]:
    """Уникальные строки поля → метка HDBSCAN (-1 шум, -2 нет эмбеддинга)."""
    empty_ph = empty_placeholder_for_field(cluster_field)
    embed_texts = [
        text_for_embedding(k, empty_placeholder=empty_ph) for k in unique_keys
    ]

    if on_progress:
        on_progress("Получение эмбеддингов…", 0.0)

    embeddings = get_embeddings_batch(embed_texts, on_progress=on_progress)

    text_to_cluster: Dict[str, int] = {}
    for k, emb in zip(unique_keys, embeddings):
        text_to_cluster[k] = -2 if emb is None else -999

    ok_pairs = [(k, e) for k, e in zip(unique_keys, embeddings) if e is not None]
    if not ok_pairs:
        for k in unique_keys:
            text_to_cluster[k] = -2
        return text_to_cluster

    if len(ok_pairs) == 1:
        only_k, _ = ok_pairs[0]
        text_to_cluster[only_k] = 0
        for k in unique_keys:
            if text_to_cluster[k] == -999:
                text_to_cluster[k] = -2
        return text_to_cluster

    keys_ok = [p[0] for p in ok_pairs]
    embeddings_array = np.array([p[1] for p in ok_pairs])
    diag = diagnose_embeddings(embeddings_array)
    if diag:
        logger.info(
            "Эмбеддинги (%s): min=%.4f max=%.4f mean=%.4f std=%.4f",
            cluster_field,
            diag["min"],
            diag["max"],
            diag["mean"],
            diag["std"],
        )

    n_ok = len(keys_ok)
    mcs = min(min_cluster_size, max(2, n_ok // 20))
    mcs = max(2, min(mcs, n_ok))
    ms_eff = max(1, min(min_samples, n_ok))

    if on_progress:
        on_progress(f"HDBSCAN (точек: {n_ok}, min_cluster_size={mcs})…", 0.85)

    use_umap_local = use_umap and umap is not None
    if use_umap and umap is None:
        logger.warning("umap-learn не установлен — используется L2-нормализация")

    if use_umap_local and n_ok <= 2:
        use_umap_local = False

    if use_umap_local:
        n_neigh = min(10, max(2, n_ok // 5), n_ok - 1)
        n_neigh = max(2, n_neigh)
        n_comp = min(50, embeddings_array.shape[1], max(2, n_ok - 1))
        init_umap = "random" if n_ok < 50 else "spectral"
        reducer = umap.UMAP(
            n_components=n_comp,
            metric="cosine",
            random_state=42,
            n_neighbors=n_neigh,
            min_dist=0.001,
            spread=0.5,
            verbose=False,
            init=init_umap,
        )
        emb_proc = reducer.fit_transform(embeddings_array)
    else:
        emb_proc = Normalizer(norm="l2").fit_transform(embeddings_array)

    clusterer = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms_eff,
        metric="euclidean",
        cluster_selection_method="eom",
        alpha=1.0,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(emb_proc)

    for k, lab in zip(keys_ok, labels):
        text_to_cluster[k] = int(lab)

    for k in unique_keys:
        if text_to_cluster[k] == -999:
            text_to_cluster[k] = -2

    return text_to_cluster


def cluster_texts_hdbscan(
    texts: List[str],
    *,
    cluster_field: str,
    umap_settings: Optional[UmapSettings] = None,
    hdbscan_settings: Optional[HdbscanSettings] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> ClusterTextsResult:
    """
    HDBSCAN по списку текстов (1:1 с записями).
    Метки: >=0 кластер, -1 шум, -2 нет эмбеддинга.
    """
    if not texts:
        return ClusterTextsResult(labels=[], viz_x=[], viz_y=[])

    max_chars = max_chars_for_cluster_field(cluster_field)
    empty_ph = empty_placeholder_for_field(cluster_field_label(cluster_field))
    embed_texts = [
        text_for_embedding(t, empty_placeholder=empty_ph, max_len=max_chars) for t in texts
    ]

    if on_progress:
        on_progress("Получение эмбеддингов…", 0.0)

    embeddings = get_embeddings_batch(
        embed_texts,
        on_progress=on_progress,
        max_chars=max_chars,
    )

    labels: List[int] = [-2] * len(texts)
    viz_x: List[Optional[float]] = [None] * len(texts)
    viz_y: List[Optional[float]] = [None] * len(texts)
    ok_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
    if not ok_indices:
        return ClusterTextsResult(
            labels=labels, viz_x=viz_x, viz_y=viz_y, embeddings=embeddings
        )

    if len(ok_indices) == 1:
        labels[ok_indices[0]] = 0
        viz_x[ok_indices[0]] = 0.0
        viz_y[ok_indices[0]] = 0.0
        return ClusterTextsResult(
            labels=labels, viz_x=viz_x, viz_y=viz_y, embeddings=embeddings
        )

    embeddings_array = np.array([embeddings[i] for i in ok_indices])
    diag = diagnose_embeddings(embeddings_array)
    if diag:
        logger.info(
            "Эмбеддинги (%s): min=%.4f max=%.4f mean=%.4f std=%.4f",
            cluster_field,
            diag["min"],
            diag["max"],
            diag["mean"],
            diag["std"],
        )

    umap_cfg = umap_settings or UmapSettings()
    hdb_cfg = hdbscan_settings or HdbscanSettings()
    n_ok = len(ok_indices)

    if on_progress:
        on_progress(f"HDBSCAN (диалогов: {n_ok})…", 0.85)

    use_umap_local = umap_cfg.enabled and umap is not None
    if umap_cfg.enabled and umap is None:
        logger.warning("umap-learn не установлен — используется L2-нормализация")
    if use_umap_local and n_ok <= 2:
        use_umap_local = False

    if use_umap_local:
        n_neigh = _resolve_umap_n_neighbors(umap_cfg.n_neighbors, n_ok)
        n_comp = _resolve_umap_n_components(
            umap_cfg.n_components, embeddings_array.shape[1], n_ok
        )
        init_umap = _resolve_umap_init(umap_cfg.init, n_ok)
        reducer = umap.UMAP(
            n_components=n_comp,
            metric=umap_cfg.metric,
            random_state=umap_cfg.random_state,
            n_neighbors=n_neigh,
            min_dist=umap_cfg.min_dist,
            spread=umap_cfg.spread,
            verbose=False,
            init=init_umap,
        )
        emb_proc = reducer.fit_transform(embeddings_array)
    else:
        emb_proc = Normalizer(norm="l2").fit_transform(embeddings_array)

    hdb_labels = _apply_hdbscan(emb_proc, n_ok=n_ok, hdbscan=hdb_cfg)

    if on_progress:
        on_progress("2D-проекция для визуализации…", 0.92)
    coords_2d = _compute_viz_coords_2d(embeddings_array, umap_cfg)

    for j, idx in enumerate(ok_indices):
        labels[idx] = int(hdb_labels[j])
        viz_x[idx] = float(coords_2d[j, 0])
        viz_y[idx] = float(coords_2d[j, 1]) if coords_2d.shape[1] > 1 else 0.0

    return ClusterTextsResult(
        labels=labels,
        viz_x=viz_x,
        viz_y=viz_y,
        embeddings=embeddings,
    )


CLUSTER_LABEL_NO_KEYWORDS = "ключевые слова не выявлены"


def derive_cluster_label(
    recs: Sequence[Mapping[str, Any]],
    *,
    cluster_field: str = "dialog",
    turn_index: int = 0,
    tfidf_words: Optional[Sequence[Tuple[str, float]]] = None,
    cid: int = -1,
) -> str:
    """Название кластера: топ-4 слова TF-IDF через «.» или запасная фраза."""
    if cid == -1:
        return "шум"

    if tfidf_words:
        words = [w for w, _ in _filter_tfidf_term_pairs(list(tfidf_words), top_n=4)]
        if words:
            return ".".join(words)

    return CLUSTER_LABEL_NO_KEYWORDS


def representative_label_for_cluster(
    recs: Sequence[Mapping[str, Any]],
    *,
    cluster_field: str = "dialog",
) -> str:
    return derive_cluster_label(recs, cluster_field=cluster_field, cid=0) or ""


def merge_cluster_tfidf_words(
    records: Sequence[Mapping[str, Any]],
    *,
    cluster_field: str,
    turn_index: int = 0,
    primary: Optional[Dict[int, List[Tuple[str, float]]]] = None,
    top_n: int = 10,
) -> Dict[int, List[Tuple[str, float]]]:
    """c-TF-IDF и запасной вариант: ClassTfidfTransformer по текстам кластера."""
    merged: Dict[int, List[Tuple[str, float]]] = dict(primary or {})
    fallback = compute_clusters_tfidf_keywords(
        records,
        cluster_field=cluster_field,
        turn_index=turn_index,
        top_n=top_n,
    )
    for cid, words in fallback.items():
        if not merged.get(cid):
            merged[cid] = words
    return merged


def build_cluster_id_to_label(
    records: List[MutableMapping[str, Any]],
    text_to_cluster: Dict[str, int],
    cluster_field: str,
) -> Dict[int, str]:
    counts: Dict[int, Counter[str]] = defaultdict(Counter)
    for r in records:
        k = field_key(r.get(cluster_field))
        cid = text_to_cluster.get(k, -2)
        if cid >= 0 and k:
            counts[cid][k] += 1
    out: Dict[int, str] = {}
    for cid, ctr in counts.items():
        if ctr:
            label, _ = ctr.most_common(1)[0]
            out[cid] = label
    return out


def build_cluster_id_to_label_from_ids(
    records: List[MutableMapping[str, Any]],
    record_cluster_ids: List[int],
    *,
    cluster_field: str = "dialog",
    turn_index: int = 0,
    cluster_tfidf_words: Optional[Dict[int, List[Tuple[str, float]]]] = None,
) -> Dict[int, str]:
    by_cid: Dict[int, List[MutableMapping[str, Any]]] = defaultdict(list)
    for rec, cid in zip(records, record_cluster_ids):
        if cid >= 0:
            by_cid[cid].append(rec)
    out: Dict[int, str] = {}
    for cid, recs in by_cid.items():
        out[cid] = derive_cluster_label(
            recs,
            cluster_field=cluster_field,
            turn_index=turn_index,
            tfidf_words=(cluster_tfidf_words or {}).get(cid, []),
            cid=cid,
        )
    if -1 in record_cluster_ids:
        out[-1] = "шум"
    return out


def compute_cluster_quality_metrics(record_cluster_ids: List[int]) -> Dict[str, Any]:
    n = len(record_cluster_ids)
    noise = sum(1 for c in record_cluster_ids if c == -1)
    sizes = Counter(c for c in record_cluster_ids if c >= 0)
    n_clusters = len(sizes)
    avg_size = sum(sizes.values()) / n_clusters if n_clusters else 0.0
    singletons = sum(1 for size in sizes.values() if size == 1)
    small_le2 = sum(1 for size in sizes.values() if size <= 2)
    return {
        "n_noise": noise,
        "noise_pct": round(100.0 * noise / n, 1) if n else 0.0,
        "avg_cluster_size": round(avg_size, 1),
        "singleton_clusters": singletons,
        "small_clusters_le2": small_le2,
    }


def assign_cluster_label(
    record: MutableMapping[str, Any],
    text_to_cluster: Dict[str, int],
    cluster_id_to_label: Dict[int, str],
    cluster_field: str,
) -> str:
    k = field_key(record.get(cluster_field))
    if not k:
        return ""
    cid = text_to_cluster.get(k, -2)
    if cid >= 0:
        return cluster_id_to_label.get(cid, k)
    return k


def normalize_field_stats(value: object) -> str:
    if value is None:
        return "(пусто)"
    s = str(value).strip()
    return s if s else "(пусто)"


def _dialog_id_from_record(rec: Mapping[str, Any]) -> str:
    for key in ("dialog_id", "id", "idx"):
        val = rec.get(key)
        if val is not None and str(val).strip():
            return str(val)
    return "?"


def format_dialog_full(rec: Mapping[str, Any]) -> str:
    """Полный текст диалога для отображения в UI (без обрезки)."""
    did = _dialog_id_from_record(rec)
    lines = [f"Диалог ID: {did}"]
    suggestion = rec.get("suggestion")
    if isinstance(suggestion, str) and suggestion.strip():
        lines.append(f"Рекомендация (suggestion): {suggestion.strip()}")
    for turn in _turns_from_record(rec):
        role = (turn.get("role") or "unknown").strip()
        content = turn.get("content")
        text = content if isinstance(content, str) else (str(content) if content else "")
        text = text.strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def format_dialog_history_only(rec: Mapping[str, Any]) -> str:
    """Только реплики из history (role: content)."""
    lines: List[str] = []
    for turn in _turns_from_record(rec):
        role = (turn.get("role") or "unknown").strip()
        text = _turn_text(turn)
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _format_dialog_for_prompt(rec: Mapping[str, Any], *, max_chars: int = 1500) -> str:
    did = _dialog_id_from_record(rec)
    lines = [f"Диалог ID: {did}"]
    suggestion = rec.get("suggestion")
    if isinstance(suggestion, str) and suggestion.strip():
        lines.append(f"Рекомендация (suggestion): {suggestion.strip()[:500]}")
    for turn in _turns_from_record(rec)[:20]:
        role = (turn.get("role") or "unknown").strip()
        content = turn.get("content")
        text = content if isinstance(content, str) else (str(content) if content else "")
        text = text.strip()[:max_chars]
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def select_llm_prompt_sample(
    records: Sequence[Mapping[str, Any]],
    rng: random.Random,
    n: int,
) -> List[Dict[str, Any]]:
    if not records:
        return []
    n = max(1, min(n, len(records)))
    if n >= len(records):
        return [dict(r) for r in records]
    return [dict(r) for r in rng.sample(list(records), n)]


DEFAULT_CLUSTER_LLM_PROMPT_TEMPLATE = """Проанализируй группу диалогов службы поддержки.

Кластер (представительная формулировка): {cluster_label}
Всего диалогов в группе: {n_records}
Примеры текста поля «{field_label}» (фрагменты):
{unique_block}

Фрагмент диалогов (выборка):
{formatted_sample}

Верни ТОЛЬКО валидный JSON:
{{
  "general_theme": "краткое название темы кластера на русском",
  "problems_summary": "связный текст о типичных проблемах группы"
}}
"""


def render_cluster_llm_prompt(
    template: str,
    *,
    cluster_label: str,
    n_records: int,
    field_label: str,
    unique_block: str,
    formatted_sample: str,
) -> str:
    ctx = {
        "cluster_label": cluster_label,
        "n_records": n_records,
        "field_label": field_label,
        "unique_block": unique_block,
        "formatted_sample": formatted_sample,
    }
    try:
        return template.format(**ctx)
    except KeyError as e:
        raise ValueError(f"Некорректный промпт LLM: неизвестная переменная {e}") from e


def run_cluster_llm_for_records(
    records: List[Dict[str, Any]],
    cluster_label: str,
    prompt_sample_records: Sequence[Mapping[str, Any]],
    *,
    unique_field: str,
    turn_index: int = 0,
    model: str,
    api_key: str = "",
    api_base: str = "",
    max_prompt_chars: int = 120_000,
    prompt_template: str = DEFAULT_CLUSTER_LLM_PROMPT_TEMPLATE,
) -> Dict[str, Any]:
    import litellm

    out: Dict[str, Any] = {
        "llm_general_theme": "",
        "problems_summary": "",
        "llm_error": "",
    }
    if not records or not prompt_sample_records:
        out["llm_error"] = "Нет записей для LLM"
        return out

    unique_vals: List[str] = []
    seen: set[str] = set()
    for r in records:
        if is_history_cluster_field(unique_field):
            v = extract_cluster_text(r, unique_field, turn_index=turn_index)
        else:
            v = field_key(r.get(unique_field))
        preview = v[:120] + ("…" if len(v) > 120 else "")
        if preview and preview not in seen:
            seen.add(preview)
            unique_vals.append(preview)

    sample_blocks = [_format_dialog_for_prompt(r) for r in prompt_sample_records]
    formatted_sample = "\n\n".join(sample_blocks)
    unique_block = "\n".join(f"- {v}" for v in unique_vals[:50]) or "(нет)"

    prompt = render_cluster_llm_prompt(
        prompt_template or DEFAULT_CLUSTER_LLM_PROMPT_TEMPLATE,
        cluster_label=cluster_label,
        n_records=len(records),
        field_label=field_scope_label(unique_field, turn_index),
        unique_block=unique_block,
        formatted_sample=formatted_sample,
    )
    if len(prompt) > max_prompt_chars:
        prompt = prompt[:max_prompt_chars] + "\n…(обрезано)"

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4000,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    try:
        response = litellm.completion(**kwargs)
        raw = (response.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            out["llm_error"] = "JSON не найден в ответе LLM"
            return out
        parsed = json.loads(match.group())
        gt = parsed.get("general_theme")
        out["llm_general_theme"] = str(gt).strip() if gt is not None else ""
        ps = parsed.get("problems_summary")
        out["problems_summary"] = str(ps).strip() if ps is not None else ""
    except Exception as e:
        out["llm_error"] = str(e)
    return out


@dataclass
class ClusteringResult:
    records: List[Dict[str, Any]]
    stats_report: str
    theme_counts: Counter[str] = field(default_factory=Counter)
    field_counts: Counter[str] = field(default_factory=Counter)
    field_theme_counts: Counter[Tuple[str, str]] = field(default_factory=Counter)
    n_clusters: int = 0
    bad_lines: int = 0
    text_to_cluster: Dict[str, int] = field(default_factory=dict)
    cluster_id_to_label: Dict[int, str] = field(default_factory=dict)
    cluster_field: str = "dialog"
    output_field: str = "cluster_label"
    embedding_diag: Dict[str, float] = field(default_factory=dict)
    viz_points: List[Dict[str, Any]] = field(default_factory=list)
    n_input_before_dedup: int = 0
    dedup_exact_removed: int = 0
    dedup_semantic_removed: int = 0
    dedup_removed_details: List[Dict[str, Any]] = field(default_factory=list)
    dedup_field: str = "dialog"
    dedup_turn_index: int = 0
    cluster_turn_index: int = 0
    cluster_tfidf_words: Dict[int, List[Tuple[str, float]]] = field(default_factory=dict)
    record_embeddings: List[Optional[List[float]]] = field(default_factory=list)
    cluster_quality: Dict[str, Any] = field(default_factory=dict)


def embeddings_by_dialog_id(
    records: Sequence[Mapping[str, Any]],
    embeddings: Sequence[Optional[List[float]]],
) -> Dict[str, Optional[List[float]]]:
    if len(records) != len(embeddings):
        return {}
    return {
        _dialog_id_from_record(rec): emb
        for rec, emb in zip(records, embeddings)
    }


def similar_dialog_indices(
    query_embedding: Sequence[float],
    embeddings: Sequence[Optional[Sequence[float]]],
    *,
    top_k: int = 10,
    min_similarity: float = 0.0,
    exclude_index: Optional[int] = None,
) -> List[Tuple[int, float]]:
    """Индексы записей, наиболее похожих на query_embedding (cosine similarity)."""
    q = np.asarray(query_embedding, dtype=float).reshape(1, -1)
    q_norm = float(np.linalg.norm(q))
    if q_norm == 0:
        return []

    q = q / q_norm
    scored: List[Tuple[int, float]] = []
    for i, emb in enumerate(embeddings):
        if i == exclude_index or emb is None:
            continue
        v = np.asarray(emb, dtype=float).reshape(1, -1)
        v_norm = float(np.linalg.norm(v))
        if v_norm == 0:
            continue
        sim = float((q @ v.T)[0, 0] / v_norm)
        if sim >= min_similarity:
            scored.append((i, sim))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[: max(1, int(top_k))]


def record_similarity_label(
    rec: Mapping[str, Any],
    *,
    output_field: str = "cluster_label",
    preview_chars: int = 72,
) -> str:
    did = _dialog_id_from_record(rec)
    cid = int(rec.get("cluster_id", -2))
    label = str(rec.get(output_field) or "").strip()
    cluster = _cluster_display_name(cid, label)
    preview = format_dialog_history_only(rec).replace("\n", " ")[:preview_chars]
    if len(format_dialog_history_only(rec)) > preview_chars:
        preview += "…"
    return f"{did} · {cluster} · {preview}"


def prepare_similarity_query_text(
    query_text: str,
    *,
    cluster_field: str,
) -> str:
    """Текст эталона для эмбеддинга — в том же формате, что при кластеризации."""
    cleaned = sanitize_cluster_text((query_text or "").strip())
    if not cleaned:
        return ""
    max_chars = max_chars_for_cluster_field(cluster_field)
    empty_ph = empty_placeholder_for_field(cluster_field_label(cluster_field))
    return text_for_embedding(cleaned, empty_placeholder=empty_ph, max_len=max_chars)


def embed_similarity_query_text(
    query_text: str,
    *,
    cluster_field: str,
) -> Optional[List[float]]:
    model_text = prepare_similarity_query_text(query_text, cluster_field=cluster_field)
    if not model_text:
        return None
    max_chars = max_chars_for_cluster_field(cluster_field)
    embs = get_embeddings_batch([model_text], max_chars=max_chars)
    return embs[0] if embs else None


def similar_dialog_search_rows_by_embedding(
    records: Sequence[Mapping[str, Any]],
    embeddings: Sequence[Optional[Sequence[float]]],
    query_embedding: Sequence[float],
    *,
    output_field: str = "cluster_label",
    top_k: int = 10,
    min_similarity: float = 0.0,
    exclude_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    hits = similar_dialog_indices(
        query_embedding,
        embeddings,
        top_k=top_k,
        min_similarity=min_similarity,
        exclude_index=exclude_index,
    )
    rows: List[Dict[str, Any]] = []
    for idx, sim in hits:
        rec = records[idx]
        cid = int(rec.get("cluster_id", -2))
        rows.append(
            {
                "similarity": round(sim, 4),
                "dialog_id": _dialog_id_from_record(rec),
                "cluster_id": cid,
                "кластер": _cluster_display_name(cid, str(rec.get(output_field) or "")),
                "превью": format_dialog_history_only(rec)[:500],
                "_index": idx,
            }
        )
    return rows


def similar_dialog_search_rows(
    records: Sequence[Mapping[str, Any]],
    embeddings: Sequence[Optional[Sequence[float]]],
    query_index: int,
    *,
    output_field: str = "cluster_label",
    top_k: int = 10,
    min_similarity: float = 0.0,
    exclude_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if query_index < 0 or query_index >= len(records):
        return []
    query_emb = embeddings[query_index] if query_index < len(embeddings) else None
    if query_emb is None:
        return []
    return similar_dialog_search_rows_by_embedding(
        records,
        embeddings,
        query_emb,
        output_field=output_field,
        top_k=top_k,
        min_similarity=min_similarity,
        exclude_index=exclude_index,
    )


def format_tfidf_words(
    words: Sequence[Tuple[str, float]],
    *,
    max_words: int = 10,
) -> str:
    parts = [f"{word} ({score:.3f})" for word, score in list(words)[:max_words] if word]
    return ", ".join(parts)


def serialize_cluster_tfidf_words(
    words: Dict[int, List[Tuple[str, float]]],
) -> Dict[str, List[List[float]]]:
    return {str(cid): [[w, s] for w, s in lst] for cid, lst in words.items()}


def parse_cluster_tfidf_words(raw: object) -> Dict[int, List[Tuple[str, float]]]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[int, List[Tuple[str, float]]] = {}
    for key, value in raw.items():
        try:
            cid = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, list):
            continue
        pairs: List[Tuple[str, float]] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((str(item[0]), float(item[1])))
        if pairs:
            out[cid] = pairs
    return out


def _keywords_for_highlight(keywords: Sequence[str]) -> List[str]:
    """Убирает короткие термы, входящие в уже выбранные более длинные n-граммы."""
    unique = sorted({k.strip() for k in keywords if k and k.strip()}, key=len, reverse=True)
    filtered: List[str] = []
    for keyword in unique:
        lowered = keyword.casefold()
        if any(lowered in existing.casefold() for existing in filtered):
            continue
        filtered.append(keyword)
    return filtered


def highlight_keywords_in_text(text: str, keywords: Sequence[str]) -> str:
    """Markdown с **жирной** подсветкой ключевых слов (биграммы — первыми)."""
    if not text or not keywords:
        return text
    unique = _keywords_for_highlight(keywords)
    if not unique:
        return text
    marked = text
    for i, keyword in enumerate(unique):
        marked = re.sub(
            re.escape(keyword),
            lambda match, idx=i: f"<<HL{idx}>>{match.group(0)}<<END{idx}>>",
            marked,
            flags=re.IGNORECASE,
        )
    for i, _keyword in enumerate(unique):
        marked = marked.replace(f"<<HL{i}>>", "**").replace(f"<<END{i}>>", "**")
    return marked


def highlight_keywords_in_html(text: str, keywords: Sequence[str]) -> str:
    """HTML-подсветка ключевых слов для таблиц (тег ``mark``)."""
    import html as html_module

    if not text:
        return ""
    escaped = html_module.escape(text)
    if not keywords:
        return escaped.replace("\n", "<br>")
    for keyword in _keywords_for_highlight(keywords):
        escaped = re.sub(
            re.escape(keyword),
            lambda match: f"<mark>{match.group(0)}</mark>",
            escaped,
            flags=re.IGNORECASE,
        )
    return escaped.replace("\n", "<br>")


def _class_tfidf_transformer():
    from bertopic.vectorizers import ClassTfidfTransformer

    return ClassTfidfTransformer(
        bm25_weighting=True,
        reduce_frequent_words=True,
    )


def _simple_class_tfidf_transformer():
    """Упрощённый c-TF-IDF для коротких текстов и малого числа тем."""
    from bertopic.vectorizers import ClassTfidfTransformer

    return ClassTfidfTransformer(
        bm25_weighting=False,
        reduce_frequent_words=False,
    )


def _count_vectorizer_for_ctfidf(
    texts: Sequence[str],
    *,
    use_stop_words: bool = True,
    max_df: float = 0.95,
) -> Tuple[Any, Any]:
    from sklearn.feature_extraction.text import CountVectorizer

    kwargs: Dict[str, Any] = {
        "ngram_range": _CTFIDF_NGRAM_RANGE,
        "min_df": 1,
        "max_df": max_df,
        "token_pattern": _VECTOR_TOKEN_PATTERN,
    }
    if use_stop_words:
        kwargs["stop_words"] = _ctfidf_stop_words()
    vectorizer = CountVectorizer(**kwargs)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def compute_clusters_tfidf_keywords(
    records: Sequence[Mapping[str, Any]],
    *,
    cluster_field: str,
    turn_index: int = 0,
    top_n: int = 10,
) -> Dict[int, List[Tuple[str, float]]]:
    """c-TF-IDF по кластерам (ClassTfidfTransformer, BM25 + reduce_frequent_words)."""
    import scipy.sparse as sp

    by_cid: Dict[int, List[str]] = defaultdict(list)
    for rec in records:
        cid = int(rec.get("cluster_id", -2))
        if cid < 0:
            continue
        text = extract_cluster_text(rec, cluster_field, turn_index=turn_index).strip()
        if text:
            by_cid[cid].append(text)

    if not by_cid:
        return {}

    all_texts: List[str] = []
    cluster_ids: List[int] = []
    for cid in sorted(by_cid.keys()):
        for text in by_cid[cid]:
            all_texts.append(text)
            cluster_ids.append(cid)

    vectorizer = None
    doc_term = None
    for use_stop_words, max_df in ((True, 0.95), (False, 0.95), (False, 1.0)):
        try:
            vectorizer, doc_term = _count_vectorizer_for_ctfidf(
                all_texts,
                use_stop_words=use_stop_words,
                max_df=max_df,
            )
            if doc_term.shape[1] > 0:
                break
        except ValueError:
            continue

    if vectorizer is None or doc_term is None or doc_term.shape[1] == 0:
        return {}

    unique_cids = sorted(set(cluster_ids))
    cid_to_row = {cid: row for row, cid in enumerate(unique_cids)}
    class_matrix = sp.lil_matrix(
        (len(unique_cids), doc_term.shape[1]),
        dtype=np.float64,
    )
    for doc_idx, cid in enumerate(cluster_ids):
        class_matrix[cid_to_row[cid]] += doc_term[doc_idx]
    class_matrix = class_matrix.tocsr()

    ctfidf_matrix = _class_tfidf_transformer().fit_transform(class_matrix)
    feature_names = vectorizer.get_feature_names_out()

    result: Dict[int, List[Tuple[str, float]]] = {}
    for cid, row_idx in cid_to_row.items():
        row = np.asarray(ctfidf_matrix[row_idx].todense()).ravel()
        order = row.argsort()[::-1]
        candidates: List[Tuple[str, float]] = []
        for idx in order:
            score = float(row[idx])
            if score <= 0:
                break
            candidates.append((str(feature_names[idx]), score))
        words = _filter_tfidf_term_pairs(candidates, top_n=top_n)
        if words:
            result[cid] = words
    return result


def build_stats_report(
    *,
    theme_counts: Counter[str],
    field_counts: Counter[str],
    field_theme_counts: Counter[Tuple[str, str]],
    total: int,
    bad: int,
    n_clusters: int,
    cluster_field: str,
    cluster_turn_index: int = 0,
    output_field: str,
    algorithm: str = "HDBSCAN",
) -> str:
    buf = StringIO()
    print(f"Всего распознанных JSON-строк: {total}", file=buf)
    if bad:
        print(f"Пропущено при парсинге: {bad}", file=buf)
    print(f"Уникальных кластеров {algorithm} (метка >= 0): {n_clusters}", file=buf)
    empty_theme = theme_counts.get("", 0)
    print(f"С полем {output_field} пустым: {empty_theme}", file=buf)
    with_theme = total - empty_theme
    pct = round(100.0 * with_theme / total, 2) if total else 0.0
    print(f"С непустым {output_field}: {with_theme} ({pct}%)", file=buf)

    print(f"\n--- По новому значению {output_field} (число диалогов) ---", file=buf)
    for theme, n in sorted(theme_counts.items(), key=lambda x: (-x[1], x[0].lower())):
        label = "(пусто)" if theme == "" else theme
        print(f"{n}\t{label}", file=buf)

    src_label = field_scope_label(cluster_field, cluster_turn_index)
    print(f"\n--- По исходному полю «{src_label}» (число диалогов) ---", file=buf)
    for sub, n in sorted(field_counts.items(), key=lambda x: (-x[1], x[0].lower())):
        print(f"{n}\t{sub}", file=buf)

    print(f"\n--- Пары «{src_label}» → {output_field} (число диалогов) ---", file=buf)
    print(f"диалогов\t{src_label}\t{output_field}", file=buf)
    for (sub, theme_l), n in sorted(
        field_theme_counts.items(),
        key=lambda x: (-x[1], x[0][0].lower(), x[0][1].lower()),
    ):
        print(f"{n}\t{sub}\t{theme_l}", file=buf)
    return buf.getvalue()


def export_jsonl_with_stats(records: List[Dict[str, Any]], stats_report: str) -> str:
    lines = [json.dumps(safe_json_write(r), ensure_ascii=False) for r in records]
    body = "\n".join(lines)
    if not (stats_report or "").strip():
        return (body + "\n") if body else ""
    stats_lines = "\n".join(f"# {ln}" if ln else "#" for ln in stats_report.splitlines())
    if body:
        body += "\n"
    return body + "\n" + stats_lines + "\n"


def _cluster_display_name(cluster_id: int, label: str) -> str:
    if cluster_id == -1:
        return "шум (-1)"
    if cluster_id == -2:
        return "нет эмбеддинга (-2)"
    name = (label or "").strip() or f"Кластер {cluster_id}"
    return f"{cluster_id}: {name}"


def cluster_size_chart_rows(
    records: List[Dict[str, Any]],
    *,
    output_field: str = "cluster_label",
) -> List[Dict[str, Any]]:
    by_cid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_cid[int(rec.get("cluster_id", -2))].append(rec)

    total = len(records)
    rows: List[Dict[str, Any]] = []
    for cid in sorted(by_cid.keys(), key=lambda c: (-len(by_cid[c]), c)):
        recs = by_cid[cid]
        label = str(recs[0].get(output_field) or "").strip() if recs else ""
        rows.append(
            {
                "cluster_id": cid,
                "кластер": _cluster_display_name(cid, label),
                "диалогов": len(recs),
                "доля, %": round(100.0 * len(recs) / total, 2) if total else 0.0,
            }
        )
    return rows


def cluster_example_rows(
    records: List[Dict[str, Any]],
    cluster_id: int,
    *,
    cluster_field: str,
    output_field: str = "cluster_label",
    limit: int = 20,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    matched = [r for r in records if int(r.get("cluster_id", -99)) == cluster_id]
    rows: List[Dict[str, Any]] = []
    for rec in matched[offset : offset + limit]:
        rows.append(
            {
                "dialog_id": _dialog_id_from_record(rec),
                "history": format_dialog_history_only(rec),
            }
        )
    return rows


@dataclass
class BalancedSampleResult:
    records: List[Dict[str, Any]]
    stats_rows: List[Dict[str, Any]]
    total: int = 0
    groups_used: int = 0


def _normalize_embedding_rows(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def mmr_select_indices(
    embeddings: np.ndarray,
    k: int,
    *,
    lambda_param: float = 0.6,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """
    Maximal Marginal Relevance: k индексов с балансом
    «типичность для кластера» (sim к центроиду) и «непохожесть на уже выбранные».
    """
    n = len(embeddings)
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))

    lam = float(lambda_param)
    lam = max(0.0, min(1.0, lam))
    rng = rng or random.Random()

    emb_norm = _normalize_embedding_rows(np.asarray(embeddings, dtype=float))
    sim_matrix = emb_norm @ emb_norm.T
    centroid = emb_norm.mean(axis=0)
    c_norm = float(np.linalg.norm(centroid))
    centroid = centroid / c_norm if c_norm > 0 else centroid
    relevance = emb_norm @ centroid

    selected: List[int] = []
    candidates = list(range(n))

    first = max(candidates, key=lambda i: (relevance[i], rng.random()))
    selected.append(first)
    candidates.remove(first)

    while len(selected) < k and candidates:
        best_idx = candidates[0]
        best_score = -float("inf")
        for i in candidates:
            max_sim_selected = max(float(sim_matrix[i, j]) for j in selected)
            score = lam * float(relevance[i]) - (1.0 - lam) * max_sim_selected
            tie = rng.random()
            if score > best_score or (score == best_score and tie > 0.5):
                best_score = score
                best_idx = i
        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected


def _select_diverse_records(
    recs: List[Dict[str, Any]],
    limit: int,
    *,
    embeddings: Optional[List[Optional[List[float]]]] = None,
    sample_method: str = "random",
    mmr_lambda: float = 0.6,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if limit >= len(recs):
        return list(recs)
    if sample_method != "mmr" or not embeddings:
        return rng.sample(recs, limit)

    ok_pairs = [(i, e) for i, e in enumerate(embeddings) if e is not None]
    if len(ok_pairs) < limit:
        return rng.sample(recs, limit)

    idx_map = [p[0] for p in ok_pairs]
    emb_array = np.array([p[1] for p in ok_pairs])
    picked_local = mmr_select_indices(
        emb_array,
        limit,
        lambda_param=mmr_lambda,
        rng=rng,
    )
    return [recs[idx_map[i]] for i in picked_local]


def build_balanced_cluster_sample(
    records: List[Dict[str, Any]],
    *,
    output_field: str = "cluster_label",
    cluster_field: str = "dialog",
    per_cluster: int = 3,
    per_noise_label: int = 1,
    include_noise: bool = True,
    total_max: Optional[int] = None,
    seed: Optional[int] = None,
    sample_method: str = "mmr",
    mmr_lambda: float = 0.6,
    record_embeddings: Optional[List[Optional[List[float]]]] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> BalancedSampleResult:
    """
    Сбалансированная выборка: до per_cluster диалогов из каждого cluster_id >= 0;
    при include_noise — до per_noise_label из каждой уникальной метки среди шума (-1).

    sample_method:
      - "random" — случайная подвыборка внутри группы;
      - "mmr" — Maximal Marginal Relevance по эмбеддингам из кластеризации
        (record_embeddings); без них — fallback на random.
    """
    if not records:
        return BalancedSampleResult(records=[], stats_rows=[])

    per_cluster = max(1, int(per_cluster))
    per_noise_label = max(1, int(per_noise_label))
    rng = random.Random(seed)
    method = (sample_method or "mmr").strip().lower()
    if method not in ("random", "mmr"):
        method = "mmr"

    by_cid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    noise: List[Dict[str, Any]] = []
    for rec in records:
        cid = int(rec.get("cluster_id", -2))
        if cid >= 0:
            by_cid[cid].append(rec)
        elif cid == -1:
            noise.append(rec)

    groups: List[Tuple[str, int, List[Dict[str, Any]]]] = []
    for cid in sorted(by_cid.keys()):
        label = str((by_cid[cid][0].get(output_field) or "")).strip() or f"Кластер {cid}"
        groups.append((label, cid, by_cid[cid]))

    noise_by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if include_noise:
        for rec in noise:
            lbl = str(rec.get(output_field) or "шум").strip() or "шум"
            noise_by_label[lbl].append(rec)
        for lbl in sorted(noise_by_label.keys(), key=str.lower):
            groups.append((lbl, -1, noise_by_label[lbl]))

    if not groups:
        return BalancedSampleResult(records=[], stats_rows=[])

    n_groups = len(groups)
    per_group = per_cluster
    if total_max is not None and total_max > 0:
        per_group = min(per_cluster, max(1, total_max // n_groups))

    group_embeddings: Dict[int, List[Optional[List[float]]]] = {}
    if method == "mmr":
        emb_by_id = embeddings_by_dialog_id(records, record_embeddings or [])
        if emb_by_id:
            if on_progress:
                on_progress("MMR: эмбеддинги из кластеризации…", 0.05)
            for gi, (_label, _cid, recs) in enumerate(groups):
                group_embeddings[gi] = [
                    emb_by_id.get(_dialog_id_from_record(rec)) for rec in recs
                ]
        else:
            logger.warning(
                "MMR: нет эмбеддингов из кластеризации (перезапустите кластеризацию) — random"
            )
            if on_progress:
                on_progress(
                    "MMR недоступен без эмбеддингов — случайная выборка…",
                    0.05,
                )

    picked: List[Dict[str, Any]] = []
    stats_rows: List[Dict[str, Any]] = []

    for gi, (label, cid, recs) in enumerate(groups):
        limit = per_noise_label if cid == -1 else per_group
        limit = min(limit, len(recs))
        if limit <= 0:
            continue
        embs = group_embeddings.get(gi) if method == "mmr" else None
        batch = _select_diverse_records(
            recs,
            limit,
            embeddings=embs,
            sample_method=method,
            mmr_lambda=mmr_lambda,
            rng=rng,
        )
        picked.extend(batch)
        stats_rows.append(
            {
                "cluster_id": cid if cid >= 0 else "шум (-1)",
                output_field: label,
                "доступно": len(recs),
                "взято": len(batch),
                "метод": method,
            }
        )

    return BalancedSampleResult(
        records=picked,
        stats_rows=stats_rows,
        total=len(picked),
        groups_used=len([r for r in stats_rows if r.get("взято")]),
    )


def run_dialog_clustering(
    records: List[Dict[str, Any]],
    *,
    cluster_field: str = "dialog",
    output_field: str = "cluster_label",
    umap_settings: Optional[UmapSettings] = None,
    hdbscan_settings: Optional[HdbscanSettings] = None,
    with_llm: bool = False,
    llm_model: str = "",
    llm_api_key: str = "",
    llm_api_base: str = "",
    llm_sample_dialogs: int = 25,
    llm_prompt_template: str = DEFAULT_CLUSTER_LLM_PROMPT_TEMPLATE,
    llm_seed: Optional[int] = None,
    dedup_enabled: bool = True,
    dedup_field: Optional[str] = None,
    dedup_turn_index: int = 0,
    cluster_turn_index: int = 0,
    dedup_similarity_threshold: float = DEFAULT_SEMANTIC_DEDUP_THRESHOLD,
    tfidf_top_n: int = 10,
    on_progress: Optional[ProgressCallback] = None,
) -> ClusteringResult:
    if not records:
        raise ValueError("Нет записей для кластеризации")

    cluster_field = validate_record_field(cluster_field)
    dedup_turn_index = max(0, int(dedup_turn_index))
    cluster_turn_index = max(0, int(cluster_turn_index))

    n_input_before_dedup = len(records)
    dedup_exact_removed = 0
    dedup_semantic_removed = 0
    dedup_removed_details: List[Dict[str, Any]] = []
    dedup_report = ""

    dedup_field_eff = validate_record_field((dedup_field or cluster_field).strip() or cluster_field)
    if not is_history_cluster_field(dedup_field_eff):
        dedup_turn_index = 0
    if not is_history_cluster_field(cluster_field):
        cluster_turn_index = 0

    if dedup_enabled:
        if on_progress:
            on_progress(
                f"Шаг 1: удаление дубликатов ({field_scope_label(dedup_field_eff, dedup_turn_index)}) "
                f"— {len(records)} диалогов…",
                0.01,
            )
        dedup = deduplicate_dialog_records(
            records,
            dedup_field=dedup_field_eff,
            dedup_turn_index=dedup_turn_index,
            similarity_threshold=dedup_similarity_threshold,
            on_progress=on_progress,
        )
        records = dedup.records
        dedup_exact_removed = dedup.n_exact_removed
        dedup_semantic_removed = dedup.n_semantic_removed
        dedup_removed_details = dedup.removed_details
        dedup_report = build_dedup_report(
            dedup,
            dedup_field=dedup_field_eff,
            dedup_turn_index=dedup_turn_index,
        )
        if not records:
            raise ValueError(
                "После удаления дубликатов не осталось диалогов для кластеризации."
            )

    cluster_texts = [
        extract_cluster_text(r, cluster_field, turn_index=cluster_turn_index) for r in records
    ]
    empty_count = sum(1 for t in cluster_texts if not t.strip())
    if empty_count == len(records):
        raise ValueError(
            f"Нет текста для кластеризации по полю «{field_scope_label(cluster_field, cluster_turn_index)}» "
            "(проверьте history в записях)."
        )

    logger.info(
        "Кластеризация: поле=%s, turn=%s, диалогов=%s, пустых=%s",
        cluster_field,
        cluster_turn_index,
        len(records),
        empty_count,
    )

    if on_progress:
        on_progress(
            f"Шаг 2: кластеризация {len(records)} диалогов по "
            f"«{field_scope_label(cluster_field, cluster_turn_index)}»…",
            0.12,
        )

    hdb_cfg = hdbscan_settings or HdbscanSettings()
    hdb_cfg = HdbscanSettings(
        min_cluster_size=max(2, hdb_cfg.min_cluster_size),
        min_samples=max(1, hdb_cfg.min_samples),
        metric=hdb_cfg.metric,
        cluster_selection_method=hdb_cfg.cluster_selection_method,
        alpha=hdb_cfg.alpha,
        auto_scale_min_cluster_size=hdb_cfg.auto_scale_min_cluster_size,
    )

    cluster_result = cluster_texts_hdbscan(
        cluster_texts,
        cluster_field=cluster_field,
        umap_settings=umap_settings,
        hdbscan_settings=hdb_cfg,
        on_progress=on_progress,
    )
    record_cluster_ids = cluster_result.labels
    viz_x = cluster_result.viz_x
    viz_y = cluster_result.viz_y

    n_clusters = len({c for c in record_cluster_ids if c >= 0})
    cluster_quality = compute_cluster_quality_metrics(record_cluster_ids)

    records_by_cid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r, cid in zip(records, record_cluster_ids):
        if cid >= 0:
            records_by_cid[cid].append(r)

    partial_records = [
        {**dict(r), "cluster_id": int(cid)}
        for r, cid in zip(records, record_cluster_ids)
    ]
    cluster_tfidf_words = compute_clusters_tfidf_keywords(
        partial_records,
        cluster_field=cluster_field,
        turn_index=cluster_turn_index,
        top_n=max(3, tfidf_top_n),
    )

    cluster_id_to_label: Dict[int, str] = {}
    for cid in set(record_cluster_ids):
        if cid == -1:
            cluster_id_to_label[cid] = "шум"
        elif cid >= 0:
            cluster_id_to_label[cid] = derive_cluster_label(
                records_by_cid.get(cid, []),
                cluster_field=cluster_field,
                turn_index=cluster_turn_index,
                tfidf_words=cluster_tfidf_words.get(cid, []),
                cid=cid,
            )

    if with_llm and llm_model:
        llm_rng = random.Random(llm_seed) if llm_seed is not None else random.Random()
        cids = sorted(records_by_cid.keys())
        for i, cid in enumerate(cids):
            recs = records_by_cid[cid]
            cluster_label = cluster_id_to_label.get(cid, "")
            if not cluster_label:
                continue
            if on_progress:
                on_progress(
                    f"LLM: кластер {i + 1}/{len(cids)} ({cluster_label[:40]}…)",
                    0.9 + 0.09 * (i + 1) / max(len(cids), 1),
                )
            sample = select_llm_prompt_sample(recs, llm_rng, max(1, llm_sample_dialogs))
            llm_out = run_cluster_llm_for_records(
                recs,
                cluster_label,
                sample,
                unique_field=cluster_field,
                turn_index=cluster_turn_index,
                model=llm_model,
                api_key=llm_api_key,
                api_base=llm_api_base,
                prompt_template=llm_prompt_template,
            )
            err = llm_out.get("llm_error") or ""
            theme = str(llm_out.get("llm_general_theme") or "").strip() or cluster_label
            if err:
                logger.warning("LLM кластер %s: %s", cid, err)
            cluster_id_to_label[cid] = theme

    theme_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    field_theme_counts: Counter[Tuple[str, str]] = Counter()
    out_records: List[Dict[str, Any]] = []
    viz_points: List[Dict[str, Any]] = []

    for obj, cid, vx, vy in zip(records, record_cluster_ids, viz_x, viz_y):
        rec = dict(obj)
        rec["cluster_id"] = int(cid)
        if cid >= 0:
            new_theme = cluster_id_to_label.get(cid, CLUSTER_LABEL_NO_KEYWORDS)
        elif cid == -1:
            new_theme = "шум"
        else:
            new_theme = ""
        rec[output_field] = new_theme
        theme_counts[new_theme] += 1

        src_text = extract_cluster_text(rec, cluster_field, turn_index=cluster_turn_index)
        preview = src_text[:80] + ("…" if len(src_text) > 80 else "")
        sub_display = normalize_field_stats(preview)
        field_counts[sub_display] += 1
        theme_lbl = "(пусто)" if new_theme == "" else new_theme
        field_theme_counts[(sub_display, theme_lbl)] += 1
        out_records.append(rec)

        if vx is not None and vy is not None:
            snippet = format_dialog_full(rec)
            viz_points.append(
                {
                    "dialog_id": _dialog_id_from_record(rec),
                    "cluster_id": int(cid),
                    "cluster_label": new_theme or "(пусто)",
                    "cluster_display": _cluster_display_name(int(cid), new_theme),
                    "viz_x": float(vx),
                    "viz_y": float(vy),
                    "snippet": snippet,
                }
            )

    stats_body = build_stats_report(
        theme_counts=theme_counts,
        field_counts=field_counts,
        field_theme_counts=field_theme_counts,
        total=len(records),
        bad=0,
        n_clusters=n_clusters,
        cluster_field=cluster_field,
        cluster_turn_index=cluster_turn_index,
        output_field=output_field,
    )
    stats_report = (dedup_report + "\n" + stats_body).strip() if dedup_report else stats_body

    return ClusteringResult(
        records=out_records,
        stats_report=stats_report,
        theme_counts=theme_counts,
        field_counts=field_counts,
        field_theme_counts=field_theme_counts,
        n_clusters=n_clusters,
        text_to_cluster={},
        cluster_id_to_label=cluster_id_to_label,
        cluster_field=cluster_field,
        output_field=output_field,
        viz_points=viz_points,
        n_input_before_dedup=n_input_before_dedup,
        dedup_exact_removed=dedup_exact_removed,
        dedup_semantic_removed=dedup_semantic_removed,
        dedup_removed_details=dedup_removed_details,
        dedup_field=dedup_field_eff,
        dedup_turn_index=dedup_turn_index,
        cluster_turn_index=cluster_turn_index,
        cluster_tfidf_words=cluster_tfidf_words,
        record_embeddings=list(cluster_result.embeddings),
        cluster_quality=cluster_quality,
    )


def cluster_summary_rows(
    records: List[Dict[str, Any]],
    *,
    cluster_field: str,
    turn_index: int = 0,
    output_field: str,
    cluster_tfidf_words: Optional[Dict[int, List[Tuple[str, float]]]] = None,
) -> List[Dict[str, Any]]:
    """Сводка по cluster_id (одна строка на кластер)."""
    by_cid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_cid[int(rec.get("cluster_id", -2))].append(rec)

    total = len(records)
    rows: List[Dict[str, Any]] = []
    src_label = field_scope_label(cluster_field, turn_index)

    def _cid_sort_key(cid: int) -> tuple:
        return (cid < 0, -len(by_cid[cid]), cid)

    for cid in sorted(by_cid.keys(), key=_cid_sort_key):
        recs = by_cid[cid]
        label = str(recs[0].get(output_field) or "").strip()
        if not label:
            label = "(пусто)"

        examples: List[str] = []
        for r in recs[:3]:
            did = _dialog_id_from_record(r)
            body = format_dialog_full(r)
            if body.strip():
                examples.append(f"#{did}:\n{body}")

        tfidf_display = "—"
        if cluster_tfidf_words and cid >= 0:
            tfidf_display = format_tfidf_words(cluster_tfidf_words.get(cid, [])) or "—"

        cid_display = str(cid) if cid >= 0 else "шум (-1)"
        rows.append(
            {
                output_field: label,
                "cluster_id": cid_display,
                "TF-IDF": tfidf_display,
                "диалогов": len(recs),
                "доля, %": round(100.0 * len(recs) / total, 2) if total else 0.0,
                f"источник ({src_label})": len({
                    extract_cluster_text(r, cluster_field, turn_index=turn_index) for r in recs
                }),
                "примеры": "\n\n---\n\n".join(examples),
            }
        )
    return rows


def _resolve_min_topic_size(settings: BertopicSettings, n_ok: int) -> int:
    mcs = settings.min_topic_size
    if settings.auto_scale_min_topic_size:
        mcs = min(mcs, max(3, n_ok // 10))
    return max(2, min(mcs, n_ok))


def _resolve_bertopic_min_df(n_ok: int) -> int:
    if n_ok < 100:
        return 1
    if n_ok < 250:
        return 2
    return 2


def _docs_for_bertopic_vectorizer(texts: Sequence[str]) -> List[str]:
    """Тексты для CountVectorizer — без placeholder, только реальный текст."""
    return [sanitize_cluster_text(t) or (t or "").strip() for t in texts]


def _bertopic_docs_stats(docs: Sequence[str]) -> Dict[str, Any]:
    non_empty = [d for d in docs if d.strip()]
    unique = len(set(non_empty))
    n = len(docs)
    avg_len = sum(len(d) for d in non_empty) / max(1, len(non_empty))
    return {
        "n_docs": n,
        "n_non_empty": len(non_empty),
        "n_unique": unique,
        "avg_len": avg_len,
        "repetitive": unique < max(3, n // 8),
        "short": avg_len < 35,
    }


def _build_bertopic_vectorizer(
    n_ok: int,
    *,
    use_stop_words: bool,
    token_pattern: Optional[str] = _VECTOR_TOKEN_PATTERN,
    min_df: Optional[int] = None,
    max_df: float = 0.95,
    ngram_range: Tuple[int, int] = _CTFIDF_NGRAM_RANGE,
) -> "CountVectorizer":
    from sklearn.feature_extraction.text import CountVectorizer

    kwargs: Dict[str, Any] = {
        "ngram_range": ngram_range,
        "min_df": 1 if min_df is None else max(1, int(min_df)),
        "max_df": float(max_df),
    }
    if token_pattern:
        kwargs["token_pattern"] = token_pattern
    if use_stop_words:
        kwargs["stop_words"] = _ctfidf_stop_words()
    return CountVectorizer(**kwargs)


def _bertopic_vectorizer_attempts(
    n_ok: int,
    docs: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    stats = _bertopic_docs_stats(docs)
    min_df_default = _resolve_bertopic_min_df(n_ok)
    permissive = stats["repetitive"] or stats["short"] or stats["n_unique"] < 5

    standard = [
        {
            "use_stop_words": True,
            "token_pattern": _VECTOR_TOKEN_PATTERN,
            "min_df": min_df_default,
            "max_df": 0.95,
            "ngram_range": _CTFIDF_NGRAM_RANGE,
        },
        {
            "use_stop_words": False,
            "token_pattern": _VECTOR_TOKEN_PATTERN,
            "min_df": 1,
            "max_df": 0.95,
            "ngram_range": _CTFIDF_NGRAM_RANGE,
        },
        {
            "use_stop_words": False,
            "token_pattern": _VECTOR_TOKEN_PATTERN,
            "min_df": 1,
            "max_df": 1.0,
            "ngram_range": _CTFIDF_NGRAM_RANGE,
        },
        {
            "use_stop_words": False,
            "token_pattern": None,
            "min_df": 1,
            "max_df": 1.0,
            "ngram_range": _CTFIDF_NGRAM_RANGE,
        },
        {
            "use_stop_words": False,
            "token_pattern": None,
            "min_df": 1,
            "max_df": 1.0,
            "ngram_range": (3, 5),
            "analyzer": "char_wb",
        },
    ]

    if permissive:
        priority = [attempt for attempt in standard if attempt.get("max_df") == 1.0]
        rest = [attempt for attempt in standard if attempt.get("max_df") != 1.0]
        attempts = priority + rest
    else:
        attempts = standard
    return attempts, stats


def _fit_bertopic_with_vectorizer_fallback(
    topic_model_factory: Callable[[Any], Any],
    docs_ok: List[str],
    embeddings_array: np.ndarray,
    *,
    topic_model_factory_variants: Optional[Sequence[Callable[[Any], Any]]] = None,
) -> Tuple[Any, List[int]]:
    attempts, stats = _bertopic_vectorizer_attempts(len(docs_ok), docs_ok)
    factories = list(topic_model_factory_variants or [topic_model_factory])
    last_error: Optional[Exception] = None
    for factory in factories:
        for attempt in attempts:
            analyzer = attempt.get("analyzer")
            vectorizer_kwargs: Dict[str, Any] = {
                "use_stop_words": attempt["use_stop_words"],
                "token_pattern": attempt.get("token_pattern"),
                "min_df": attempt.get("min_df"),
                "max_df": attempt.get("max_df", 0.95),
                "ngram_range": attempt.get("ngram_range", (1, 2)),
            }
            if analyzer == "char_wb":
                from sklearn.feature_extraction.text import CountVectorizer

                vectorizer_model = CountVectorizer(
                    analyzer="char_wb",
                    ngram_range=attempt.get("ngram_range", (3, 5)),
                    min_df=1,
                    max_df=1.0,
                )
            else:
                vectorizer_model = _build_bertopic_vectorizer(
                    len(docs_ok),
                    **vectorizer_kwargs,
                )
            try:
                topic_model = factory(vectorizer_model)
                topics, _ = topic_model.fit_transform(docs_ok, embeddings=embeddings_array)
                return topic_model, list(topics)
            except (ValueError, TypeError) as exc:
                last_error = exc
                msg = str(exc).casefold()
                if isinstance(exc, TypeError) or "empty vocabulary" in msg or "max_df" in msg or "min_df" in msg:
                    logger.warning(
                        "Кластеризация fit (stop_words=%s, max_df=%s, ngram=%s, analyzer=%s): %s",
                        attempt["use_stop_words"],
                        attempt.get("max_df"),
                        attempt.get("ngram_range"),
                        analyzer or "word",
                        exc,
                    )
                    continue
                raise
    if last_error is not None:
        hint = (
            "Попробуйте «Все фразы» вместо одной реплики, поле «Диалог» "
            "или увеличьте выборку."
        )
        if stats["n_non_empty"] < 3:
            hint = (
                "Слишком мало непустых текстов по выбранному полю/реплике. "
                "Проверьте history или выберите «Все фразы»."
            )
        elif stats["repetitive"]:
            hint = (
                "Тексты почти одинаковые (типично для «Только 1-я фраза»). "
                "Выберите «Все фразы» или поле «Диалог»."
            )
        raise ValueError(
            "Кластеризация не смогла построить словарь тем по текстам диалогов "
            f"(после всех fallback: документов {stats['n_docs']}, "
            f"уникальных {stats['n_unique']}, средняя длина {stats['avg_len']:.0f}). "
            f"{hint}"
        ) from last_error
    raise ValueError("Кластеризация: не удалось выполнить fit_transform")


def _bertopic_topic_label(topic_model: Any, topic_id: int, *, top_n: int = 5) -> str:
    if topic_id == -1:
        return "шум"
    if topic_id == -2:
        return ""
    words = topic_model.get_topic(topic_id) or []
    labels = [word for word, _ in _filter_tfidf_term_pairs(words, top_n=top_n)]
    return ", ".join(labels) if labels else f"Тема {topic_id}"


def _bertopic_topic_tfidf_words(
    topic_model: Any,
    topic_id: int,
    *,
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    if topic_id < 0:
        return []
    words = topic_model.get_topic(topic_id) or []
    return _filter_tfidf_term_pairs(
        [(str(word), float(score)) for word, score in words],
        top_n=top_n,
    )


@dataclass
class BertopicClusterTextsResult:
    labels: List[int]
    viz_x: List[Optional[float]]
    viz_y: List[Optional[float]]
    topic_labels: Dict[int, str] = field(default_factory=dict)
    topic_keywords: Dict[int, str] = field(default_factory=dict)
    topic_tfidf_words: Dict[int, List[Tuple[str, float]]] = field(default_factory=dict)
    embeddings: List[Optional[List[float]]] = field(default_factory=list)


def _hdbscan_settings_from_bertopic(cfg: BertopicSettings, n_ok: int) -> HdbscanSettings:
    mcs = _resolve_min_topic_size(cfg, n_ok)
    return HdbscanSettings(
        min_cluster_size=mcs,
        min_samples=max(1, min(cfg.min_samples, n_ok)),
        metric=cfg.hdbscan_metric,
        cluster_selection_method=cfg.cluster_selection_method,
        alpha=cfg.alpha,
        auto_scale_min_cluster_size=cfg.auto_scale_min_topic_size,
    )


def cluster_texts_bertopic(
    texts: List[str],
    *,
    cluster_field: str,
    umap_settings: Optional[UmapSettings] = None,
    bertopic_settings: Optional[BertopicSettings] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> BertopicClusterTextsResult:
    """
    Кластеризация по списку текстов (1:1 с записями) с готовыми эмбеддингами.
    Метки: >=0 тема, -1 шум, -2 нет эмбеддинга.
    """
    if not texts:
        return BertopicClusterTextsResult(labels=[], viz_x=[], viz_y=[])

    try:
        from bertopic import BERTopic
    except ImportError as e:
        raise ImportError(
            "Для кластеризации установите пакет: pip install bertopic"
        ) from e

    max_chars = max_chars_for_cluster_field(cluster_field)
    empty_ph = empty_placeholder_for_field(cluster_field_label(cluster_field))
    model_texts = [
        text_for_cluster_model(t, empty_placeholder=empty_ph, max_len=max_chars)
        for t in texts
    ]

    if on_progress:
        on_progress("Получение эмбеддингов…", 0.0)

    embeddings = get_embeddings_batch(
        model_texts,
        on_progress=on_progress,
        max_chars=max_chars,
    )

    labels: List[int] = [-2] * len(texts)
    viz_x: List[Optional[float]] = [None] * len(texts)
    viz_y: List[Optional[float]] = [None] * len(texts)
    ok_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
    if not ok_indices:
        return BertopicClusterTextsResult(
            labels=labels, viz_x=viz_x, viz_y=viz_y, embeddings=embeddings
        )

    if len(ok_indices) == 1:
        labels[ok_indices[0]] = 0
        viz_x[ok_indices[0]] = 0.0
        viz_y[ok_indices[0]] = 0.0
        return BertopicClusterTextsResult(
            labels=labels,
            viz_x=viz_x,
            viz_y=viz_y,
            topic_labels={0: texts[ok_indices[0]][:80] or "Тема 0"},
            topic_keywords={0: texts[ok_indices[0]][:120] or "Тема 0"},
            topic_tfidf_words={0: [(texts[ok_indices[0]][:80] or "Тема 0", 1.0)]},
            embeddings=embeddings,
        )

    embeddings_array = np.array([embeddings[i] for i in ok_indices])
    diag = diagnose_embeddings(embeddings_array)
    if diag:
        logger.info(
            "Эмбеддинги кластеризации (%s): min=%.4f max=%.4f mean=%.4f std=%.4f",
            cluster_field,
            diag["min"],
            diag["max"],
            diag["mean"],
            diag["std"],
        )

    umap_cfg = umap_settings or UmapSettings()
    bertopic_cfg = bertopic_settings or BertopicSettings()
    n_ok = len(ok_indices)

    if not bertopic_cfg.use_ctfidf:
        if on_progress:
            on_progress(f"Кластеризация без c-TF-IDF (диалогов: {n_ok})…", 0.85)
        hdb_cfg = _hdbscan_settings_from_bertopic(bertopic_cfg, n_ok)
        use_umap_only = umap_cfg.enabled and umap is not None and n_ok > 2
        if umap_cfg.enabled and umap is None:
            logger.warning("umap-learn не установлен — L2-нормализация")
        if use_umap_only:
            n_neigh = _resolve_umap_n_neighbors(umap_cfg.n_neighbors, n_ok)
            n_comp = _resolve_umap_n_components(
                umap_cfg.n_components, embeddings_array.shape[1], n_ok
            )
            init_umap = _resolve_umap_init(umap_cfg.init, n_ok)
            reducer = umap.UMAP(
                n_components=n_comp,
                metric=umap_cfg.metric,
                random_state=umap_cfg.random_state,
                n_neighbors=n_neigh,
                min_dist=umap_cfg.min_dist,
                spread=umap_cfg.spread,
                verbose=False,
                init=init_umap,
            )
            emb_proc = reducer.fit_transform(embeddings_array)
        else:
            emb_proc = Normalizer(norm="l2").fit_transform(embeddings_array)
        hdb_labels = _apply_hdbscan(emb_proc, n_ok=n_ok, hdbscan=hdb_cfg)
        if on_progress:
            on_progress("2D-проекция для визуализации…", 0.92)
        coords_2d = _compute_viz_coords_2d(embeddings_array, umap_cfg)
        for j, idx in enumerate(ok_indices):
            labels[idx] = int(hdb_labels[j])
            viz_x[idx] = float(coords_2d[j, 0])
            viz_y[idx] = float(coords_2d[j, 1]) if coords_2d.shape[1] > 1 else 0.0
        topic_labels: Dict[int, str] = {}
        topic_keywords: Dict[int, str] = {}
        for cid in set(labels):
            if cid == -1:
                topic_labels[cid] = "шум"
                topic_keywords[cid] = "шум"
            elif cid >= 0:
                topic_labels[cid] = f"Тема {cid}"
                topic_keywords[cid] = f"Тема {cid}"
        return BertopicClusterTextsResult(
            labels=labels,
            viz_x=viz_x,
            viz_y=viz_y,
            topic_labels=topic_labels,
            topic_keywords=topic_keywords,
            topic_tfidf_words={},
            embeddings=embeddings,
        )

    if on_progress:
        on_progress(f"Кластеризация (диалогов: {n_ok})…", 0.85)

    mcs = _resolve_min_topic_size(bertopic_cfg, n_ok)
    ms_eff = max(1, min(bertopic_cfg.min_samples, n_ok))

    use_umap_local = umap_cfg.enabled and umap is not None
    if umap_cfg.enabled and umap is None:
        logger.warning("umap-learn не установлен — кластеризация без UMAP")
    if use_umap_local and n_ok <= 2:
        use_umap_local = False

    umap_model = None
    if use_umap_local:
        n_neigh = _resolve_umap_n_neighbors(umap_cfg.n_neighbors, n_ok)
        n_comp = _resolve_umap_n_components(
            umap_cfg.n_components, embeddings_array.shape[1], n_ok
        )
        init_umap = _resolve_umap_init(umap_cfg.init, n_ok)
        umap_model = umap.UMAP(
            n_components=n_comp,
            metric=umap_cfg.metric,
            random_state=umap_cfg.random_state,
            n_neighbors=n_neigh,
            min_dist=umap_cfg.min_dist,
            spread=umap_cfg.spread,
            verbose=False,
            init=init_umap,
        )

    hdbscan_model = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms_eff,
        metric=bertopic_cfg.hdbscan_metric,
        cluster_selection_method=bertopic_cfg.cluster_selection_method,
        alpha=bertopic_cfg.alpha,
        prediction_data=True,
    )

    docs_ok = [_docs_for_bertopic_vectorizer(texts)[i] for i in ok_indices]

    def _make_topic_model(vectorizer_model: Any, *, simple_ctfidf: bool = False) -> Any:
        ctfidf = (
            _simple_class_tfidf_transformer()
            if simple_ctfidf
            else _class_tfidf_transformer()
        )
        return BERTopic(
            embedding_model=None,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf,
            top_n_words=max(3, bertopic_cfg.top_n_words),
            calculate_probabilities=bertopic_cfg.calculate_probabilities,
            verbose=False,
        )

    def _factory_standard(vectorizer_model: Any) -> Any:
        return _make_topic_model(vectorizer_model, simple_ctfidf=False)

    def _factory_simple(vectorizer_model: Any) -> Any:
        return _make_topic_model(vectorizer_model, simple_ctfidf=True)

    topic_model, topics = _fit_bertopic_with_vectorizer_fallback(
        _factory_standard,
        docs_ok,
        embeddings_array,
        topic_model_factory_variants=[_factory_standard, _factory_simple],
    )

    if bertopic_cfg.reduce_outliers and any(t == -1 for t in topics):
        if on_progress:
            on_progress("Кластеризация: reduce_outliers…", 0.9)
        topics = topic_model.reduce_outliers(
            docs_ok,
            topics,
            strategy=bertopic_cfg.outlier_strategy,
            embeddings=embeddings_array,
        )
        topic_model.update_topics(docs_ok, topics=topics)

    if bertopic_cfg.reduce_topics:
        if on_progress:
            on_progress("Кластеризация: reduce_topics…", 0.93)
        nr = bertopic_cfg.nr_topics
        if nr == "auto":
            topic_model.reduce_topics(docs_ok, nr_topics="auto")
        else:
            topic_model.reduce_topics(docs_ok, nr_topics=max(2, int(nr)))
        topics = list(topic_model.topics_)

    topic_labels: Dict[int, str] = {}
    topic_keywords: Dict[int, str] = {}
    topic_tfidf_words: Dict[int, List[Tuple[str, float]]] = {}
    top_n_tfidf = max(10, bertopic_cfg.top_n_words)
    for tid in set(topics):
        tid_int = int(tid)
        kw = _bertopic_topic_label(
            topic_model, tid_int, top_n=bertopic_cfg.top_n_words
        )
        topic_labels[tid_int] = kw
        topic_keywords[tid_int] = kw
        topic_tfidf_words[tid_int] = _bertopic_topic_tfidf_words(
            topic_model,
            tid_int,
            top_n=top_n_tfidf,
        )

    if on_progress:
        on_progress("2D-проекция для визуализации…", 0.95)
    coords_2d = _compute_viz_coords_2d(embeddings_array, umap_cfg)

    for j, idx in enumerate(ok_indices):
        labels[idx] = int(topics[j])
        viz_x[idx] = float(coords_2d[j, 0])
        viz_y[idx] = float(coords_2d[j, 1]) if coords_2d.shape[1] > 1 else 0.0

    return BertopicClusterTextsResult(
        labels=labels,
        viz_x=viz_x,
        viz_y=viz_y,
        topic_labels=topic_labels,
        topic_keywords=topic_keywords,
        topic_tfidf_words=topic_tfidf_words,
        embeddings=embeddings,
    )


def run_dialog_bertopic_clustering(
    records: List[Dict[str, Any]],
    *,
    cluster_field: str = "dialog",
    output_field: str = "cluster_label",
    umap_settings: Optional[UmapSettings] = None,
    bertopic_settings: Optional[BertopicSettings] = None,
    with_llm: bool = False,
    llm_model: str = "",
    llm_api_key: str = "",
    llm_api_base: str = "",
    llm_sample_dialogs: int = 25,
    llm_prompt_template: str = DEFAULT_CLUSTER_LLM_PROMPT_TEMPLATE,
    llm_seed: Optional[int] = None,
    dedup_enabled: bool = True,
    dedup_field: Optional[str] = None,
    dedup_turn_index: int = 0,
    cluster_turn_index: int = 0,
    dedup_similarity_threshold: float = DEFAULT_SEMANTIC_DEDUP_THRESHOLD,
    tfidf_top_n: int = 10,
    on_progress: Optional[ProgressCallback] = None,
) -> ClusteringResult:
    if not records:
        raise ValueError("Нет записей для кластеризации")

    cluster_field = validate_record_field(cluster_field)
    dedup_turn_index = max(0, int(dedup_turn_index))
    cluster_turn_index = max(0, int(cluster_turn_index))

    n_input_before_dedup = len(records)
    dedup_exact_removed = 0
    dedup_semantic_removed = 0
    dedup_removed_details: List[Dict[str, Any]] = []
    dedup_report = ""

    dedup_field_eff = validate_record_field((dedup_field or cluster_field).strip() or cluster_field)
    if not is_history_cluster_field(dedup_field_eff):
        dedup_turn_index = 0
    if not is_history_cluster_field(cluster_field):
        cluster_turn_index = 0

    if dedup_enabled:
        if on_progress:
            on_progress(
                f"Шаг 1: удаление дубликатов ({field_scope_label(dedup_field_eff, dedup_turn_index)}) "
                f"— {len(records)} диалогов…",
                0.01,
            )
        dedup = deduplicate_dialog_records(
            records,
            dedup_field=dedup_field_eff,
            dedup_turn_index=dedup_turn_index,
            similarity_threshold=dedup_similarity_threshold,
            on_progress=on_progress,
        )
        records = dedup.records
        dedup_exact_removed = dedup.n_exact_removed
        dedup_semantic_removed = dedup.n_semantic_removed
        dedup_removed_details = dedup.removed_details
        dedup_report = build_dedup_report(
            dedup,
            dedup_field=dedup_field_eff,
            dedup_turn_index=dedup_turn_index,
        )
        if not records:
            raise ValueError(
                "После удаления дубликатов не осталось диалогов для кластеризации."
            )

    cluster_texts = [
        extract_cluster_text(r, cluster_field, turn_index=cluster_turn_index) for r in records
    ]
    empty_count = sum(1 for t in cluster_texts if not t.strip())
    if empty_count == len(records):
        raise ValueError(
            f"Нет текста для кластеризации по полю «{field_scope_label(cluster_field, cluster_turn_index)}» "
            "(проверьте history в записях)."
        )

    logger.info(
        "Кластеризация: поле=%s, turn=%s, диалогов=%s, пустых=%s",
        cluster_field,
        cluster_turn_index,
        len(records),
        empty_count,
    )

    if on_progress:
        on_progress(
            f"Шаг 2: кластеризация {len(records)} диалогов по "
            f"«{field_scope_label(cluster_field, cluster_turn_index)}»…",
            0.12,
        )

    bertopic_cfg = bertopic_settings or BertopicSettings()
    bertopic_cfg = BertopicSettings(
        min_topic_size=max(2, bertopic_cfg.min_topic_size),
        min_samples=max(1, bertopic_cfg.min_samples),
        hdbscan_metric=bertopic_cfg.hdbscan_metric,
        cluster_selection_method=bertopic_cfg.cluster_selection_method,
        alpha=bertopic_cfg.alpha,
        auto_scale_min_topic_size=bertopic_cfg.auto_scale_min_topic_size,
        reduce_outliers=bertopic_cfg.reduce_outliers,
        outlier_strategy=bertopic_cfg.outlier_strategy,
        reduce_topics=bertopic_cfg.reduce_topics,
        nr_topics=bertopic_cfg.nr_topics,
        top_n_words=max(3, bertopic_cfg.top_n_words),
        calculate_probabilities=bertopic_cfg.calculate_probabilities,
        use_ctfidf=bertopic_cfg.use_ctfidf,
    )

    cluster_result = cluster_texts_bertopic(
        cluster_texts,
        cluster_field=cluster_field,
        umap_settings=umap_settings,
        bertopic_settings=bertopic_cfg,
        on_progress=on_progress,
    )
    record_cluster_ids = cluster_result.labels
    viz_x = cluster_result.viz_x
    viz_y = cluster_result.viz_y

    n_clusters = len({c for c in record_cluster_ids if c >= 0})
    cluster_quality = compute_cluster_quality_metrics(record_cluster_ids)

    records_by_cid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r, cid in zip(records, record_cluster_ids):
        if cid >= 0:
            records_by_cid[cid].append(r)

    cluster_tfidf_words: Dict[int, List[Tuple[str, float]]] = {}
    if bertopic_cfg.use_ctfidf:
        cluster_tfidf_words = merge_cluster_tfidf_words(
            [
                {**dict(rec), "cluster_id": int(cid)}
                for rec, cid in zip(records, record_cluster_ids)
            ],
            cluster_field=cluster_field,
            turn_index=cluster_turn_index,
            primary=cluster_result.topic_tfidf_words,
            top_n=max(3, tfidf_top_n),
        )

    cluster_id_to_label: Dict[int, str] = {}
    for cid in set(record_cluster_ids):
        if cid == -1:
            cluster_id_to_label[cid] = "шум"
        elif cid >= 0:
            cluster_id_to_label[cid] = (
                cluster_result.topic_labels.get(cid)
                or derive_cluster_label(
                    records_by_cid.get(cid, []),
                    cluster_field=cluster_field,
                    turn_index=cluster_turn_index,
                    tfidf_words=cluster_tfidf_words.get(cid, []),
                    cid=cid,
                )
            )

    if with_llm and llm_model:
        llm_rng = random.Random(llm_seed) if llm_seed is not None else random.Random()
        cids = sorted(records_by_cid.keys())
        for i, cid in enumerate(cids):
            recs = records_by_cid[cid]
            cluster_label = cluster_id_to_label.get(cid, "")
            if not cluster_label:
                continue
            if on_progress:
                on_progress(
                    f"LLM: тема {i + 1}/{len(cids)} ({cluster_label[:40]}…)",
                    0.9 + 0.09 * (i + 1) / max(len(cids), 1),
                )
            sample = select_llm_prompt_sample(recs, llm_rng, max(1, llm_sample_dialogs))
            llm_out = run_cluster_llm_for_records(
                recs,
                cluster_label,
                sample,
                unique_field=cluster_field,
                turn_index=cluster_turn_index,
                model=llm_model,
                api_key=llm_api_key,
                api_base=llm_api_base,
                prompt_template=llm_prompt_template,
            )
            err = llm_out.get("llm_error") or ""
            theme = str(llm_out.get("llm_general_theme") or "").strip() or cluster_label
            if err:
                logger.warning("LLM кластеризация тема %s: %s", cid, err)
            cluster_id_to_label[cid] = theme

    theme_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    field_theme_counts: Counter[Tuple[str, str]] = Counter()
    out_records: List[Dict[str, Any]] = []
    viz_points: List[Dict[str, Any]] = []

    for obj, cid, vx, vy in zip(records, record_cluster_ids, viz_x, viz_y):
        rec = dict(obj)
        rec["cluster_id"] = int(cid)
        if cid >= 0:
            new_theme = cluster_id_to_label.get(cid, CLUSTER_LABEL_NO_KEYWORDS)
            tfidf_words = cluster_tfidf_words.get(cid, [])
            rec["topic_keywords"] = (
                format_tfidf_words(tfidf_words) if tfidf_words else "—"
            )
        elif cid == -1:
            new_theme = "шум"
            rec["topic_keywords"] = "шум"
        else:
            new_theme = ""
            rec["topic_keywords"] = ""
        rec[output_field] = new_theme
        theme_counts[new_theme] += 1

        src_text = extract_cluster_text(rec, cluster_field, turn_index=cluster_turn_index)
        preview = src_text[:80] + ("…" if len(src_text) > 80 else "")
        sub_display = normalize_field_stats(preview)
        field_counts[sub_display] += 1
        theme_lbl = "(пусто)" if new_theme == "" else new_theme
        field_theme_counts[(sub_display, theme_lbl)] += 1
        out_records.append(rec)

        if vx is not None and vy is not None:
            snippet = format_dialog_full(rec)
            viz_points.append(
                {
                    "dialog_id": _dialog_id_from_record(rec),
                    "cluster_id": int(cid),
                    "cluster_label": new_theme or "(пусто)",
                    "cluster_display": _cluster_display_name(int(cid), new_theme),
                    "viz_x": float(vx),
                    "viz_y": float(vy),
                    "snippet": snippet,
                }
            )

    stats_body = build_stats_report(
        theme_counts=theme_counts,
        field_counts=field_counts,
        field_theme_counts=field_theme_counts,
        total=len(records),
        bad=0,
        n_clusters=n_clusters,
        cluster_field=cluster_field,
        cluster_turn_index=cluster_turn_index,
        output_field=output_field,
        algorithm=CLUSTERING_ALGORITHM_LABEL,
    )
    stats_report = (dedup_report + "\n" + stats_body).strip() if dedup_report else stats_body

    return ClusteringResult(
        records=out_records,
        stats_report=stats_report,
        theme_counts=theme_counts,
        field_counts=field_counts,
        field_theme_counts=field_theme_counts,
        n_clusters=n_clusters,
        cluster_id_to_label=cluster_id_to_label,
        cluster_field=cluster_field,
        output_field=output_field,
        viz_points=viz_points,
        n_input_before_dedup=n_input_before_dedup,
        dedup_exact_removed=dedup_exact_removed,
        dedup_semantic_removed=dedup_semantic_removed,
        dedup_removed_details=dedup_removed_details,
        dedup_field=dedup_field_eff,
        dedup_turn_index=dedup_turn_index,
        cluster_turn_index=cluster_turn_index,
        cluster_tfidf_words=cluster_tfidf_words,
        record_embeddings=list(cluster_result.embeddings),
        cluster_quality=cluster_quality,
    )
