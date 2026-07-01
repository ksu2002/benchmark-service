"""Точка входа в приложение."""

from contextlib import asynccontextmanager

import torch
import uvicorn
from api.api import router
from fastapi import FastAPI

import core.logger_config as logger_config
from core.config import settings
from rag_embedder.models.gemma import GemmaEmbedderDeployment


def _create_app() -> FastAPI:
    """Создает и настраивает экземпляр FastAPI приложения.

    Returns:
        Настроенное FastAPI приложение с моделью машинного обучения.

    Notes:
        - Использует асинхронный контекстный менеджер для управления жизненным циклом приложения
        - Инициализирует модель при запуске приложения
        - Очищает видеопамять GPU при завершении работы приложения

    """

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        app.state.model = GemmaEmbedderDeployment()
        if not app.state.model.check_health():
            raise RuntimeError("Model failed health check!")
        yield
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    app = FastAPI(
        lifespan=_lifespan, title="Эмбеддер текста. Используется в RAG сервисах.", redoc_url=None
    )
    app.include_router(router)
    return app


app = _create_app()


if __name__ == "__main__":
    uvicorn.run(
        app="main:app",
        host="0.0.0.0",
        port=8080,
        log_level=settings.log_level,
        log_config=logger_config.LOGGING_CONFIG,
        reload=settings.debug,
    )
