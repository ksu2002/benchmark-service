"""Тесты текстовых утилит кластеризации."""

from __future__ import annotations

import unittest

from tests import test_bootstrap  # noqa: F401
from clustering.text import (
    extract_cluster_text,
    load_jsonl_records,
    sanitize_cluster_text,
    validate_record_field,
)


class ClusteringTextTests(unittest.TestCase):
    """Проверяет чистые функции без внешних API."""

    def test_sanitize_cluster_text_removes_nullish_tokens(self) -> None:
        cleaned = sanitize_cluster_text("none Привет,true")

        self.assertEqual(cleaned, "Привет,")

    def test_extract_cluster_text_filters_roles(self) -> None:
        record = {
            "history": [
                {"role": "user", "content": "Хочу узнать график"},
                {"role": "assistant", "content": "Офис работает до 18:00"},
            ]
        }

        user_text = extract_cluster_text(record, "user")
        assistant_text = extract_cluster_text(record, "assistant")

        self.assertEqual(user_text, "Хочу узнать график")
        self.assertEqual(assistant_text, "Офис работает до 18:00")

    def test_validate_record_field_rejects_reserved_name(self) -> None:
        with self.assertRaises(ValueError):
            validate_record_field("history")

    def test_load_jsonl_records_counts_bad_lines(self) -> None:
        records, bad = load_jsonl_records('{"a": 1}\nnot-json\n{"b": 2}')

        self.assertEqual(len(records), 2)
        self.assertEqual(bad, 1)


if __name__ == "__main__":
    unittest.main()
