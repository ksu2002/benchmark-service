"""Тесты общих JSONL-утилит."""

from __future__ import annotations

import unittest

from tests import test_bootstrap  # noqa: F401
from common.jsonl import iter_jsonl_objects, parse_jsonl_text_to_dialog_groups, records_to_jsonl


class CommonJsonlTests(unittest.TestCase):
    """Проверяет базовый JSONL IO."""

    def test_iter_jsonl_objects_skips_comments_and_blank_lines(self) -> None:
        text = '\n# comment\n{"a": 1}\n\n{"b": 2}\n'

        result = iter_jsonl_objects(text)

        self.assertEqual(result, [(3, {"a": 1}), (5, {"b": 2})])

    def test_records_to_jsonl_serializes_multiple_records(self) -> None:
        text = records_to_jsonl([{"a": 1}, {"b": "x"}])

        self.assertEqual(text, '{"a":1}\n{"b":"x"}')

    def test_parse_jsonl_text_to_dialog_groups_builds_timeline_and_turns(self) -> None:
        text = (
            '{"dialog_id":"d-1","scenario_id":"s-1","goals":["goal"],'
            '"history":['
            '{"role":"user","content":"привет"},'
            '{"role":"assistant","content":"шум"},'
            '{"role":"assistant","content":"ответ"}]}'
        )

        groups, turns_df = parse_jsonl_text_to_dialog_groups(
            text,
            is_noise=lambda content: content == "шум",
            extract_tool_calls=lambda msg: [],
            seen_dialog_ids=set(),
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["dialog_id"], "d-1")
        self.assertEqual(
            groups[0]["timeline"],
            [
                {"role": "user", "content": "привет", "type": "message"},
                {"role": "assistant", "content": "ответ", "type": "message"},
            ],
        )
        self.assertEqual(len(turns_df), 2)
        self.assertListEqual(turns_df["dialog_id"].tolist(), ["d-1", "d-1"])


if __name__ == "__main__":
    unittest.main()

