"""Схема запроса."""

from pydantic import BaseModel, Field


class TextEmbedderRequest(BaseModel):
    """Запрос для обращения к эмбеддеру текста."""

    text: str = Field(
        description="Текст.",
        examples=("У меня не работает интернет", "Серебрянная ложка"),
    )
