"""Основные тесты чистой логики бенчмарка."""

from __future__ import annotations

import unittest

from tests import test_bootstrap  # noqa: F401
from benchmarking.core import normalize_parsed_case, parse_benchmark_cases_jsonl_text
from benchmarking.page_utils import load_test_cases_from_jsonl_text


class BenchmarkingCoreTests(unittest.TestCase):
    """Проверяет парсинг и нормализацию кейсов бенчмарка."""

    def test_normalize_parsed_case_parses_json_string_context(self) -> None:
        case = {"context": '{"city": "Челябинск"}'}

        normalized = normalize_parsed_case(case)

        self.assertEqual(normalized["context"], {"city": "Челябинск"})

    def test_parse_benchmark_cases_jsonl_text_normalizes_goals_context_and_dialog_id(self) -> None:
        text = "\n".join(
            [
                '{"goals":"Узнать адрес","context":{"city":"Челябинск"}}',
                '{"goals":["Проверить баланс"],"context":{"region":"74"},"dialog_id":"dlg-1"}',
            ]
        )

        cases, error = parse_benchmark_cases_jsonl_text(text)

        self.assertIsNone(error)
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0]["goals"], ["Узнать адрес"])
        self.assertEqual(cases[0]["context"], {"city": "Челябинск"})
        self.assertTrue(cases[0]["dialog_id"])
        self.assertEqual(cases[1]["goals"], ["Проверить баланс"])
        self.assertEqual(cases[1]["context"], {"region": "74"})
        self.assertEqual(cases[1]["dialog_id"], "dlg-1")

    def test_parse_benchmark_cases_jsonl_text_returns_error_for_invalid_context(self) -> None:
        cases, error = parse_benchmark_cases_jsonl_text(
            '{"goals":"test","context":["not","dict"]}'
        )

        self.assertEqual(cases, [])
        self.assertIn("context", error or "")

    def test_load_test_cases_from_jsonl_text_generates_missing_dialog_id(self) -> None:
        cases = load_test_cases_from_jsonl_text(
            '{"goals":"Уточнить график","context":{"branch":"центр"}}'
        )

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["goals"], ["Уточнить график"])
        self.assertEqual(cases[0]["context"], {"branch": "центр"})
        self.assertTrue(cases[0]["dialog_id"])


if __name__ == "__main__":
    unittest.main()
