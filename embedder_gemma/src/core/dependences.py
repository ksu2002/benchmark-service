"""Получение состояния приложения."""

from fastapi import Request


def get_model(request: Request):
    """Получает экземпляр модели из состояния приложения.

    Args:
        request: Объект запроса FastAPI.

    Returns:
        Модель для вычисления эмбеддингов.

    """
    return request.app.state.model
