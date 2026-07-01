"""Парное сравнение двух прогонов бенчмарка по dialog_id."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scipy.stats import binomtest

from benchmarking.bootstrap import benchmark_accuracy_values, bootstrap_ci


def _row_pass(row: dict) -> bool:
    return float(row.get("accuracy", 0) or 0) >= 1.0


def _dialog_id(row: dict) -> Optional[str]:
    did = row.get("dialog_id")
    if did is None:
        return None
    s = str(did).strip()
    return s or None


def _index_results(results: Sequence[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for row in results:
        if not isinstance(row, dict):
            continue
        did = _dialog_id(row)
        if did:
            out[did] = row
    return out


def _extract_category(row: dict) -> str:
    ctx = row.get("context")
    if isinstance(ctx, dict):
        cat = ctx.get("category")
        if cat is not None and str(cat).strip():
            return str(cat).strip()
    return "(без category)"


def _format_goals(row: dict) -> str:
    gt = row.get("goals_text")
    if isinstance(gt, str) and gt.strip():
        return gt.strip()
    g = row.get("goals")
    if isinstance(g, str) and g.strip():
        return g.strip()
    if isinstance(g, list) and g:
        return ", ".join(str(x) for x in g)
    return "Без цели"


def _extract_reason(row: dict) -> str:
    reason = row.get("reason")
    if reason is None:
        return ""
    return str(reason).strip()


def _change_type(pass_a: bool, pass_b: bool) -> str:
    if pass_a and pass_b:
        return "same_pass"
    if not pass_a and not pass_b:
        return "same_fail"
    if not pass_a and pass_b:
        return "improved"
    return "regressed"


def _mcnemar_p(improved: int, regressed: int) -> Optional[float]:
    n = improved + regressed
    if n == 0:
        return None
    k = min(improved, regressed)
    return float(binomtest(k, n, 0.5, alternative="two-sided").pvalue)


def _accuracy_ci(results: Sequence[dict]) -> Tuple[Optional[float], Optional[float]]:
    bi = bootstrap_ci(
        benchmark_accuracy_values(results),
        confidence=0.95,
        n_bootstrap=5000,
        seed=42,
    )
    if bi is None:
        return None, None
    return bi.ci_low, bi.ci_high


def _mean_accuracy(results: Sequence[dict]) -> Optional[float]:
    vals = benchmark_accuracy_values(results)
    if not vals:
        return None
    return sum(vals) / len(vals)


@dataclass(frozen=True)
class PairedCase:
    dialog_id: str
    pass_a: bool
    pass_b: bool
    change: str
    category: str
    goals: str
    reason_a: str
    reason_b: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dialog_id": self.dialog_id,
            "pass_a": self.pass_a,
            "pass_b": self.pass_b,
            "change": self.change,
            "category": self.category,
            "goals": self.goals,
            "reason_a": self.reason_a,
            "reason_b": self.reason_b,
        }


@dataclass
class RunCompareResult:
    n_paired: int
    n_only_a: int
    n_only_b: int
    acc_a: Optional[float]
    acc_b: Optional[float]
    acc_delta_paired: Optional[float]
    mcnemar_p: Optional[float]
    acc_a_ci: Tuple[Optional[float], Optional[float]]
    acc_b_ci: Tuple[Optional[float], Optional[float]]
    both_pass: int
    both_fail: int
    improved: int
    regressed: int
    paired: List[PairedCase]

    def category_breakdown(self) -> List[Dict[str, Any]]:
        groups: Dict[str, List[PairedCase]] = defaultdict(list)
        for p in self.paired:
            groups[p.category].append(p)

        rows: List[Dict[str, Any]] = []
        for category in sorted(groups.keys()):
            cases = groups[category]
            n = len(cases)
            pass_a = sum(1 for c in cases if c.pass_a)
            pass_b = sum(1 for c in cases if c.pass_b)
            rate_a = pass_a / n if n else None
            rate_b = pass_b / n if n else None
            delta = (rate_b - rate_a) if rate_a is not None and rate_b is not None else None
            rows.append(
                {
                    "category": category,
                    "n": n,
                    "acc_a": pass_a,
                    "acc_b": pass_b,
                    "rate_a": rate_a,
                    "rate_b": rate_b,
                    "delta": delta,
                    "improved": sum(1 for c in cases if c.change == "improved"),
                    "regressed": sum(1 for c in cases if c.change == "regressed"),
                }
            )
        return rows


def compare_runs(results_a: Sequence[dict], results_b: Sequence[dict]) -> RunCompareResult:
    """Сравнивает два списка строк results.jsonl по dialog_id."""
    idx_a = _index_results(results_a)
    idx_b = _index_results(results_b)

    ids_a = set(idx_a.keys())
    ids_b = set(idx_b.keys())
    common = sorted(ids_a & ids_b)

    paired: List[PairedCase] = []
    both_pass = both_fail = improved = regressed = 0

    for did in common:
        row_a = idx_a[did]
        row_b = idx_b[did]
        pass_a = _row_pass(row_a)
        pass_b = _row_pass(row_b)
        change = _change_type(pass_a, pass_b)

        if change == "same_pass":
            both_pass += 1
        elif change == "same_fail":
            both_fail += 1
        elif change == "improved":
            improved += 1
        else:
            regressed += 1

        paired.append(
            PairedCase(
                dialog_id=did,
                pass_a=pass_a,
                pass_b=pass_b,
                change=change,
                category=_extract_category(row_a) if _extract_category(row_a) != "(без category)" else _extract_category(row_b),
                goals=_format_goals(row_a),
                reason_a=_extract_reason(row_a),
                reason_b=_extract_reason(row_b),
            )
        )

    n_paired = len(paired)
    acc_delta_paired: Optional[float] = None
    if n_paired:
        acc_delta_paired = (
            sum(1 for p in paired if p.pass_b) / n_paired
            - sum(1 for p in paired if p.pass_a) / n_paired
        )

    return RunCompareResult(
        n_paired=n_paired,
        n_only_a=len(ids_a - ids_b),
        n_only_b=len(ids_b - ids_a),
        acc_a=_mean_accuracy(results_a),
        acc_b=_mean_accuracy(results_b),
        acc_delta_paired=acc_delta_paired,
        mcnemar_p=_mcnemar_p(improved, regressed),
        acc_a_ci=_accuracy_ci(results_a),
        acc_b_ci=_accuracy_ci(results_b),
        both_pass=both_pass,
        both_fail=both_fail,
        improved=improved,
        regressed=regressed,
        paired=paired,
    )
