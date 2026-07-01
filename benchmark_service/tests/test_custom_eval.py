"""Тесты безопасного custom_eval."""

from __future__ import annotations

import unittest

from benchmarking.evaluation.custom_eval import evaluate_custom_eval_code


class CustomEvalTests(unittest.TestCase):
    """Проверяет безопасный вычислитель пользовательских критериев."""

    def test_default_compare_goal_expression(self) -> None:
        result = evaluate_custom_eval_code(
            'goals.strip().lower() in str(eval_value).strip().lower()',
            {
                "goals": "Адрес офиса",
                "eval_value": "Адрес офиса: ул. Ленина",
                "response": {},
                "history": [],
                "context": {},
            },
        )
        self.assertTrue(result)

    def test_nested_response_get(self) -> None:
        result = evaluate_custom_eval_code(
            'response.get("address", {}).get("status") == "full"',
            {
                "goals": "",
                "eval_value": "",
                "response": {"address": {"status": "full"}},
                "history": [],
                "context": {},
            },
        )
        self.assertTrue(result)

    def test_rejects_import(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_custom_eval_code("__import__('os').system('echo')", {"goals": "x"})

    def test_rejects_unknown_name(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_custom_eval_code("open('/etc/passwd')", {"goals": "x"})

    def test_empty_code_returns_default(self) -> None:
        self.assertFalse(evaluate_custom_eval_code("", {"goals": "x"}, default=False))
        self.assertTrue(evaluate_custom_eval_code("", {"goals": "x"}, default=True))


if __name__ == "__main__":
    unittest.main()
