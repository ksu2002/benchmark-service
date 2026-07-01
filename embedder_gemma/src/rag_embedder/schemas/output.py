"""Схема ответа."""

from pydantic import BaseModel, Field


class TextEmbedderResponse(BaseModel):
    """Ответ от эмбеддера."""

    embedding: list[float] = Field(
        description="Векторизированный текст. Размерность вектора 768.",
        examples=[0.9, 0.1],
        min_length=768,
    )
