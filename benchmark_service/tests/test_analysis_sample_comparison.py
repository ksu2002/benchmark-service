"""Тесты аналитики сравнения выборок."""

from __future__ import annotations

import unittest

from tests import test_bootstrap  # noqa: F401
from analysis.sample_comparison import (
    category_coverage_pct,
    compare_samples,
    compute_sample_metrics,
    mean_pairwise_cosine_similarity,
)


class SampleComparisonTests(unittest.TestCase):
    """Проверяет ключевые метрики сравнения выборок."""

    def test_category_coverage_pct_counts_overlap(self) -> None:
        coverage = category_coverage_pct({"a": 2, "b": 1}, {"a": 3, "c": 1})

        self.assertEqual(coverage, 50.0)

    def test_mean_pairwise_cosine_similarity_returns_expected_value(self) -> None:
        similarity = mean_pairwise_cosine_similarity(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        )

        self.assertAlmostEqual(similarity or 0.0, 1.0 / 3.0, places=6)

    def test_compute_sample_metrics_counts_goals_turns_and_clusters(self) -> None:
        records = [
            {"goals": ["g1"], "history": [{"role": "user", "content": "1"}], "cluster_id": 1},
            {"goals": ["g2"], "history": [{"role": "user", "content": "1"}, {"role": "assistant", "content": "2"}], "cluster_id": 1},
            {"goals": ["g1"], "history": [], "cluster_id": 2},
        ]

        metrics = compute_sample_metrics(records)

        self.assertEqual(metrics.n_dialogs, 3)
        self.assertEqual(metrics.n_turns, 3)
        self.assertAlmostEqual(metrics.mean_turns, 1.0)
        self.assertEqual(metrics.n_goals, 2)
        self.assertEqual(metrics.n_clusters, 2)

    def test_compare_samples_builds_non_empty_rows(self) -> None:
        baseline = [
            {"dialog_id": "1", "goals": ["g1"], "history": [], "cluster_id": 1},
            {"dialog_id": "2", "goals": ["g2"], "history": [], "cluster_id": 2},
        ]
        current = [
            {"dialog_id": "3", "goals": ["g1"], "history": [{"role": "user", "content": "hi"}], "cluster_id": 1},
        ]

        comparison = compare_samples(baseline, current)

        self.assertEqual(comparison.goal_coverage_pct, 50.0)
        self.assertTrue(comparison.comparison_rows())


if __name__ == "__main__":
    unittest.main()
