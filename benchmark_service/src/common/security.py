"""Утилиты безопасного отображения данных."""

from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet

REDACT_PLACEHOLDER = "[скрыто]"

_REDACT_KEY_NORMALIZED: FrozenSet[str] = frozenset(
    {
        "secretaccesskey",
        "accesskey",
        "secretkey",
        "awsaccesskeyid",
        "awssecretaccesskey",
        "apikey",
        "password",
        "authorization",
        "token",
        "refreshtoken",
        "privatekey",
        "minioaccesskey",
        "miniosecretkey",
    }
)


def normalize_key_name(name: str) -> str:
    """Нормализует имя ключа для сравнения с allow/deny-списками.

    Аргументы:
        name: Исходное имя ключа.

    Возвращает:
        Имя в нижнем регистре без символов кроме латиницы и цифр.
    """

    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def redact_secrets_for_display(obj: Any, *, max_depth: int = 48) -> Any:
    """Рекурсивно скрывает чувствительные поля перед выводом в UI.

    Аргументы:
        obj: Объект для безопасного отображения.
        max_depth: Максимальная глубина рекурсии.

    Возвращает:
        Копия объекта, где секретные значения заменены на ``[скрыто]``.
    """

    if max_depth <= 0:
        return "<max_depth>"
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for key, value in obj.items():
            if normalize_key_name(key) in _REDACT_KEY_NORMALIZED:
                out[key] = REDACT_PLACEHOLDER
            else:
                out[key] = redact_secrets_for_display(value, max_depth=max_depth - 1)
        return out
    if isinstance(obj, list):
        return [redact_secrets_for_display(x, max_depth=max_depth - 1) for x in obj]
    if isinstance(obj, tuple):
        return tuple(redact_secrets_for_display(x, max_depth=max_depth - 1) for x in obj)
    return obj

