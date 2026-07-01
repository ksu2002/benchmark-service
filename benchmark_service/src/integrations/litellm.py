"""Клиентские утилиты для LiteLLM-compatible API."""

from __future__ import annotations

import os

import requests

LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE")


def get_model_names() -> list[str]:
    """Возвращает список моделей из LiteLLM ``/models``.

    При отсутствии настроек или ошибке API функция возвращает запасную модель и
    не бросает исключение, чтобы UI мог стартовать без доступного LiteLLM.

    Возвращает:
        Список имён моделей в формате ``owned_by/id`` или запасная модель.
    """

    fallback = [os.getenv("LITELLM_MODEL_NAME", "openai/gpt-4o-mini")]
    if not (LITELLM_API_BASE and str(LITELLM_API_BASE).strip()):
        return fallback
    if not (LITELLM_API_KEY and str(LITELLM_API_KEY).strip()):
        return fallback
    url = f"{str(LITELLM_API_BASE).rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {LITELLM_API_KEY}"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json().get("data") or []
        names = [
            f"{m['owned_by']}/{m['id']}"
            for m in data
            if isinstance(m, dict) and "owned_by" in m and "id" in m
        ]
        return names if names else fallback
    except Exception:
        return fallback
