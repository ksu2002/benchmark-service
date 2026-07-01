"""Общие функции форматирования времени."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

DISPLAY_TZ_UTC_PLUS_5 = timezone(timedelta(hours=5))


def format_datetime_utc_plus_5(value: Any) -> str:
    """Переводит момент времени в UTC+5 и форматирует для UI.

    Аргументы:
        value: ``datetime``, ISO-строка или произвольное значение.

    Возвращает:
        Строковое представление времени для отображения.
    """

    if value is None:
        return "—"
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local = dt.astimezone(DISPLAY_TZ_UTC_PLUS_5)
        return local.strftime("%Y-%m-%d %H:%M:%S") + " (UTC+5)"
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return "—"
        try:
            normalized = s.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            return format_datetime_utc_plus_5(dt)
        except ValueError:
            return s
    return str(value)

