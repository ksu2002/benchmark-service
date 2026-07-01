"""Совместимый модуль разбора JSONL для страниц разметки.

Новый общий код находится в ``common.jsonl``.
"""

from __future__ import annotations

from typing import Callable, Tuple

import pandas as pd

from common.jsonl import parse_jsonl_text_to_dialog_groups as _parse_dialog_groups
from common.utils import extract_tool_calls_from_jsonl_history_message


def parse_jsonl_text_to_dialog_groups(
    text: str,
    *,
    is_noise: Callable[[str], bool],
    seen_dialog_ids: set | None = None,
) -> Tuple[list, pd.DataFrame]:
    """Преобразует JSONL с историями диалогов в группы для страниц разметки.

    Аргументы:
        text: Содержимое JSONL.
        is_noise: Предикат, который отбрасывает шумовые реплики.
        seen_dialog_ids: Опциональное множество для накопления встреченных id.

    Возвращает:
        Кортеж из списка групп диалогов и DataFrame с репликами.
    """

    return _parse_dialog_groups(
        text,
        is_noise=is_noise,
        extract_tool_calls=extract_tool_calls_from_jsonl_history_message,
        seen_dialog_ids=seen_dialog_ids,
    )
