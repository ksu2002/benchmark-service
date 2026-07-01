"""Bootstrap-доверительные интервалы для метрик бенчмарка."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from benchmarking.runner import (
    llm_eval_scoring_field_names,
    parse_llm_eval_fields,
    row_criterion_accuracy,
    row_mean_criterion_score,
)


@dataclass(frozen=True)
class BootstrapInterval:
    """Результат bootstrap-оценки с доверительным интервалом."""

    estimate: float
    ci_low: float
    ci_high: float
    n: int
    confidence: float
    n_bootstrap: int


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: Optional[int] = 42,
) -> Optional[BootstrapInterval]:
    """Считает bootstrap по перцентилям для среднего значения.

    Аргументы:
        values: Наблюдения по диалогам.
        n_bootstrap: Число bootstrap-выборок.
        confidence: Уровень доверия для интервала.
        seed: Начальное значение генератора случайных чисел.

    Возвращает:
        Интервал bootstrap или ``None``, если данных нет.
    """

    clean = [float(v) for v in values]
    if not clean:
        return None
    arr = np.asarray(clean, dtype=float)
    n = int(arr.size)
    point = float(arr.mean())
    if n == 1:
        return BootstrapInterval(
            estimate=point,
            ci_low=point,
            ci_high=point,
            n=n,
            confidence=confidence,
            n_bootstrap=0,
        )

    n_boot = max(100, int(n_bootstrap))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = arr[idx].mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    ci_low = float(np.percentile(boots, 100.0 * alpha))
    ci_high = float(np.percentile(boots, 100.0 * (1.0 - alpha)))
    return BootstrapInterval(
        estimate=point,
        ci_low=ci_low,
        ci_high=ci_high,
        n=n,
        confidence=confidence,
        n_bootstrap=n_boot,
    )


def benchmark_accuracy_values(results: Sequence[dict]) -> List[float]:
    """Извлекает значения accuracy из результатов бенчмарка."""

    return [float(r.get("accuracy", 0)) for r in results if isinstance(r, dict)]


def benchmark_accuracy_bootstrap(
    results: Sequence[dict],
    **kwargs,
) -> Optional[BootstrapInterval]:
    """Считает bootstrap для accuracy по набору результатов."""

    return bootstrap_ci(benchmark_accuracy_values(results), **kwargs)


def benchmark_criterion_bootstrap_cis(
    results: Sequence[dict],
    llm_eval_fields: str = "",
    **kwargs,
) -> Dict[str, BootstrapInterval]:
    """Считает bootstrap-интервалы по каждому критерию LLM-оценки."""

    scoring = llm_eval_scoring_field_names(parse_llm_eval_fields(llm_eval_fields))
    out: Dict[str, BootstrapInterval] = {}
    for field in scoring:
        vals: List[float] = []
        for row in results:
            if not isinstance(row, dict):
                continue
            per_row = row_criterion_accuracy(row, scoring)
            if field in per_row:
                vals.append(float(per_row[field]))
        ci = bootstrap_ci(vals, **kwargs)
        if ci is not None:
            out[field] = ci
    return out


def benchmark_mean_criterion_score_bootstrap(
    results: Sequence[dict],
    llm_eval_fields: str = "",
    **kwargs,
) -> Optional[BootstrapInterval]:
    """Считает bootstrap для среднего балла по критериям."""

    scoring = llm_eval_scoring_field_names(parse_llm_eval_fields(llm_eval_fields))
    if not scoring:
        return None
    vals: List[float] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        m = row_mean_criterion_score(row, scoring)
        if m is not None:
            vals.append(float(m))
    return bootstrap_ci(vals, **kwargs)


def format_bootstrap_interval(
    interval: BootstrapInterval,
    *,
    as_percent: bool = True,
) -> str:
    """Форматирует bootstrap-интервал для отображения."""

    if as_percent:
        return (
            f"{interval.estimate:.2%} "
            f"[{interval.ci_low:.2%} – {interval.ci_high:.2%}]"
        )
    return (
        f"{interval.estimate:.4f} "
        f"[{interval.ci_low:.4f} – {interval.ci_high:.4f}]"
    )


def format_bootstrap_ci_caption(
    interval: BootstrapInterval,
    *,
    confidence: float,
) -> str:
    """Формирует подпись с уровнем доверия и размером выборки."""

    conf_pct = int(round(float(confidence) * 100))
    return (
        f"{conf_pct}% ДИ: {interval.ci_low:.2%} – {interval.ci_high:.2%} · n={interval.n}"
    )
