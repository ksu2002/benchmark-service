"""Общие функции чтения и записи JSONL."""

from __future__ import annotations

import json
from typing import Any, Callable, Tuple

import pandas as pd


def iter_jsonl_objects(text: str) -> list[tuple[int, dict]]:
    """Разбирает текст JSONL в список JSON-объектов.

    Аргументы:
        text: Содержимое JSONL-файла.

    Возвращает:
        Список пар ``(номер_строки, объект)`` для непустых строк.

    Исключения:
        ValueError: Если строка не является JSON-объектом.
    """

    out: list[tuple[int, dict]] = []
    for line_num, raw_line in enumerate((text or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Некорректная строка {line_num}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Строка {line_num}: ожидается JSON-объект")
        out.append((line_num, item))
    return out


def records_to_jsonl(records: list[Any], *, compact: bool = True) -> str:
    """Сериализует список записей в JSONL.

    Аргументы:
        records: Записи для сериализации.
        compact: Использовать компактные разделители без лишних пробелов.

    Возвращает:
        Текст JSONL.
    """

    separators = (",", ":") if compact else None
    return "\n".join(
        json.dumps(record, ensure_ascii=False, separators=separators)
        for record in records
    )


def parse_jsonl_text_to_dialog_groups(
    text: str,
    *,
    is_noise: Callable[[str], bool],
    extract_tool_calls: Callable[[dict], list],
    seen_dialog_ids: set | None = None,
) -> Tuple[list, pd.DataFrame]:
    """Преобразует JSONL с историями диалогов в группы для страниц разметки.

    Аргументы:
        text: Содержимое JSONL.
        is_noise: Предикат, который отбрасывает шумовые реплики.
        extract_tool_calls: Функция извлечения tool calls из сообщения истории.
        seen_dialog_ids: Опциональное множество для накопления встреченных id.

    Возвращает:
        Кортеж из списка групп диалогов и DataFrame с репликами.
    """

    dialog_groups = []
    for _, item in iter_jsonl_objects(text):
        history = item.get("history", [])
        scenario_id = item.get("scenario_id", "")
        goals = item.get("goals", [""])
        if isinstance(goals, str):
            goals = [goals]
        timeline = []
        for msg in history:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content", "")
            tool_calls = extract_tool_calls(msg)
            if tool_calls:
                timeline.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": tool_calls,
                        "type": "tool_call",
                    }
                )
            else:
                if is_noise(str(content or "")):
                    continue
                timeline.append({"role": role, "content": content, "type": "message"})
        if not timeline:
            continue
        dialog_id = item.get("dialog_id", "")
        dialog_groups.append(
            {
                "dialog_id": dialog_id,
                "scenario_id": scenario_id,
                "timeline": timeline,
                "original_goals": goals,
            }
        )
        if dialog_id and seen_dialog_ids is not None:
            seen_dialog_ids.add(dialog_id)

    records = []
    for item in dialog_groups:
        for msg in item["timeline"]:
            if msg["type"] == "message" and str(msg["content"]).strip():
                records.append(
                    {
                        "dialog_id": item["dialog_id"],
                        "role": msg["role"],
                        "content": msg["content"],
                    }
                )
    turns_df = (
        pd.DataFrame(records)
        if records
        else pd.DataFrame(columns=["dialog_id", "role", "content"])
    )
    return dialog_groups, turns_df

