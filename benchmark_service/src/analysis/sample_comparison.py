"""Метрики сравнения выборок: баланс, покрытие, разнообразие."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from clustering.text import _dialog_id_from_record, embeddings_by_dialog_id


def goal_from_record(rec: Mapping[str, Any]) -> str:
    goals = rec.get("goals") or []
    if goals:
        label = str(goals[0]).strip()
        return label or "Без цели"
    return "Без цели"


def turn_count_from_record(rec: Mapping[str, Any]) -> int:
    history = rec.get("history")
    if isinstance(history, list):
        return len(history)
    return 0


def cluster_id_from_record(rec: Mapping[str, Any]) -> Optional[int]:
    if "cluster_id" not in rec:
        return None
    try:
        return int(rec.get("cluster_id", -2))
    except (TypeError, ValueError):
        return None


def normalized_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = [c / total for c in counts.values() if c > 0]
    n = len(probs)
    if n <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in probs)
    h_max = math.log(n)
    return h / h_max if h_max > 0 else 0.0


def gini_coefficient(counts: Counter) -> float:
    values = sorted(c for c in counts.values() if c > 0)
    n = len(values)
    if n == 0:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    gini_sum = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(values))
    return gini_sum / (n * total)


def max_min_ratio(counts: Counter) -> Optional[float]:
    positive = [c for c in counts.values() if c > 0]
    if len(positive) < 2:
        return None
    return max(positive) / min(positive)


def category_coverage_pct(
    baseline_counts: Counter,
    current_counts: Counter,
) -> Optional[float]:
    baseline_keys = {k for k, v in baseline_counts.items() if v > 0}
    if not baseline_keys:
        return None
    current_keys = {k for k, v in current_counts.items() if v > 0}
    return 100.0 * len(baseline_keys & current_keys) / len(baseline_keys)


def mean_pairwise_cosine_similarity(
    embeddings: Sequence[Optional[Sequence[float]]],
    *,
    max_pairs: int = 2000,
    seed: int = 42,
) -> Optional[float]:
    normed: List[np.ndarray] = []
    for emb in embeddings:
        if emb is None:
            continue
        v = np.asarray(emb, dtype=float)
        nrm = float(np.linalg.norm(v))
        if nrm <= 0:
            continue
        normed.append(v / nrm)
    n = len(normed)
    if n < 2:
        return None

    all_pairs = n * (n - 1) // 2
    if all_pairs <= max_pairs:
        sims = [
            float(np.dot(normed[i], normed[j]))
            for i in range(n)
            for j in range(i + 1, n)
        ]
        return sum(sims) / len(sims)

    rng = random.Random(seed)
    sims = [
        float(np.dot(normed[i], normed[j]))
        for i, j in (rng.sample(range(n), 2) for _ in range(max_pairs))
    ]
    return sum(sims) / len(sims)


def embeddings_for_records(
    records: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
    source_embeddings: Sequence[Optional[Sequence[float]]],
) -> List[Optional[List[float]]]:
    by_id = embeddings_by_dialog_id(source_records, source_embeddings)
    return [by_id.get(_dialog_id_from_record(r)) for r in records]


@dataclass
class SampleMetrics:
    n_dialogs: int
    n_turns: int
    mean_turns: float
    n_goals: int
    goal_entropy: float
    goal_gini: float
    goal_max_min: Optional[float]
    n_clusters: int
    cluster_entropy: Optional[float]
    cluster_gini: Optional[float]
    cluster_max_min: Optional[float]
    mean_pairwise_similarity: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_dialogs": self.n_dialogs,
            "n_turns": self.n_turns,
            "mean_turns": round(self.mean_turns, 2),
            "n_goals": self.n_goals,
            "goal_entropy": round(self.goal_entropy, 3),
            "goal_gini": round(self.goal_gini, 3),
            "goal_max_min": (
                round(self.goal_max_min, 2) if self.goal_max_min is not None else None
            ),
            "n_clusters": self.n_clusters,
            "cluster_entropy": (
                round(self.cluster_entropy, 3)
                if self.cluster_entropy is not None
                else None
            ),
            "cluster_gini": (
                round(self.cluster_gini, 3) if self.cluster_gini is not None else None
            ),
            "cluster_max_min": (
                round(self.cluster_max_min, 2)
                if self.cluster_max_min is not None
                else None
            ),
            "mean_pairwise_similarity": (
                round(self.mean_pairwise_similarity, 4)
                if self.mean_pairwise_similarity is not None
                else None
            ),
        }


def compute_sample_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    record_embeddings: Optional[Sequence[Optional[Sequence[float]]]] = None,
    embedding_source_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> SampleMetrics:
    n = len(records)
    turn_counts = [turn_count_from_record(r) for r in records]
    n_turns = sum(turn_counts)
    mean_turns = n_turns / n if n else 0.0

    goal_counts: Counter = Counter(goal_from_record(r) for r in records)
    cluster_counts: Counter = Counter()
    for r in records:
        cid = cluster_id_from_record(r)
        if cid is not None and cid >= 0:
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

    cluster_entropy: Optional[float] = None
    cluster_gini: Optional[float] = None
    cluster_max_min: Optional[float] = None
    if cluster_counts:
        cluster_entropy = normalized_entropy(cluster_counts)
        cluster_gini = gini_coefficient(cluster_counts)
        cluster_max_min = max_min_ratio(cluster_counts)

    mean_sim: Optional[float] = None
    if record_embeddings is not None:
        src = embedding_source_records if embedding_source_records is not None else records
        embs = embeddings_for_records(records, src, record_embeddings)
        mean_sim = mean_pairwise_cosine_similarity(embs)

    return SampleMetrics(
        n_dialogs=n,
        n_turns=n_turns,
        mean_turns=mean_turns,
        n_goals=len(goal_counts),
        goal_entropy=normalized_entropy(goal_counts),
        goal_gini=gini_coefficient(goal_counts),
        goal_max_min=max_min_ratio(goal_counts),
        n_clusters=len(cluster_counts),
        cluster_entropy=cluster_entropy,
        cluster_gini=cluster_gini,
        cluster_max_min=cluster_max_min,
        mean_pairwise_similarity=mean_sim,
    )


@dataclass
class SampleComparison:
    baseline_label: str
    current_label: str
    baseline: SampleMetrics
    current: SampleMetrics
    goal_coverage_pct: Optional[float]
    cluster_coverage_pct: Optional[float]

    def comparison_rows(self) -> List[Dict[str, Any]]:
        b = self.baseline
        c = self.current

        def delta(cur: Optional[float], base: Optional[float], *, invert: bool = False) -> str:
            if cur is None or base is None:
                return "—"
            d = cur - base
            if abs(d) < 1e-9:
                return "0"
            better = (d > 0) if not invert else (d < 0)
            sign = "+" if d > 0 else ""
            arrow = "↑" if better else "↓"
            if isinstance(cur, int):
                return f"{sign}{d} {arrow}"
            return f"{sign}{d:.3f} {arrow}"

        rows: List[Dict[str, Any]] = [
            {"Метрика": "Диалогов", self.baseline_label: b.n_dialogs, self.current_label: c.n_dialogs, "Δ": delta(c.n_dialogs, b.n_dialogs)},
            {"Метрика": "Средняя длина (реплик)", self.baseline_label: b.mean_turns, self.current_label: c.mean_turns, "Δ": delta(c.mean_turns, b.mean_turns)},
            {"Метрика": "Энтропия по целям (0–1, ↑ лучше)", self.baseline_label: b.goal_entropy, self.current_label: c.goal_entropy, "Δ": delta(c.goal_entropy, b.goal_entropy)},
            {"Метрика": "Джини по целям (↓ лучше)", self.baseline_label: b.goal_gini, self.current_label: c.goal_gini, "Δ": delta(c.goal_gini, b.goal_gini, invert=True)},
            {"Метрика": "Max/min по целям (↓ лучше)", self.baseline_label: b.goal_max_min if b.goal_max_min is not None else "—", self.current_label: c.goal_max_min if c.goal_max_min is not None else "—", "Δ": delta(c.goal_max_min, b.goal_max_min, invert=True)},
            {"Метрика": "Покрытие целей, %", self.baseline_label: "100", self.current_label: round(self.goal_coverage_pct, 1) if self.goal_coverage_pct is not None else "—", "Δ": "—"},
        ]
        if b.n_clusters or c.n_clusters:
            rows.extend([
                {"Метрика": "Кластеров (id ≥ 0)", self.baseline_label: b.n_clusters, self.current_label: c.n_clusters, "Δ": delta(c.n_clusters, b.n_clusters)},
                {"Метрика": "Энтропия по кластерам (↑ лучше)", self.baseline_label: b.cluster_entropy if b.cluster_entropy is not None else "—", self.current_label: c.cluster_entropy if c.cluster_entropy is not None else "—", "Δ": delta(c.cluster_entropy, b.cluster_entropy)},
                {"Метрика": "Джини по кластерам (↓ лучше)", self.baseline_label: b.cluster_gini if b.cluster_gini is not None else "—", self.current_label: c.cluster_gini if c.cluster_gini is not None else "—", "Δ": delta(c.cluster_gini, b.cluster_gini, invert=True)},
                {"Метрика": "Покрытие кластеров, %", self.baseline_label: "100", self.current_label: round(self.cluster_coverage_pct, 1) if self.cluster_coverage_pct is not None else "—", "Δ": "—"},
            ])
        if b.mean_pairwise_similarity is not None or c.mean_pairwise_similarity is not None:
            rows.append({"Метрика": "Средняя pairwise similarity (↓ лучше)", self.baseline_label: b.mean_pairwise_similarity if b.mean_pairwise_similarity is not None else "—", self.current_label: c.mean_pairwise_similarity if c.mean_pairwise_similarity is not None else "—", "Δ": delta(c.mean_pairwise_similarity, b.mean_pairwise_similarity, invert=True)})
        return rows


def compare_samples(
    baseline_records: Sequence[Mapping[str, Any]],
    current_records: Sequence[Mapping[str, Any]],
    *,
    baseline_label: str = "Исходная",
    current_label: str = "Текущая",
    record_embeddings: Optional[Sequence[Optional[Sequence[float]]]] = None,
    embedding_source_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> SampleComparison:
    baseline_goals = Counter(goal_from_record(r) for r in baseline_records)
    current_goals = Counter(goal_from_record(r) for r in current_records)
    baseline_clusters: Counter = Counter()
    current_clusters: Counter = Counter()
    for r in baseline_records:
        cid = cluster_id_from_record(r)
        if cid is not None and cid >= 0:
            baseline_clusters[cid] += 1
    for r in current_records:
        cid = cluster_id_from_record(r)
        if cid is not None and cid >= 0:
            current_clusters[cid] += 1
    emb_src = embedding_source_records or baseline_records
    return SampleComparison(
        baseline_label=baseline_label,
        current_label=current_label,
        baseline=compute_sample_metrics(baseline_records, record_embeddings=record_embeddings, embedding_source_records=emb_src),
        current=compute_sample_metrics(current_records, record_embeddings=record_embeddings, embedding_source_records=emb_src),
        goal_coverage_pct=category_coverage_pct(baseline_goals, current_goals),
        cluster_coverage_pct=category_coverage_pct(baseline_clusters, current_clusters),
    )
