"""Эндпоинт для генерации текстовых эмбеддингов."""

from fastapi import APIRouter
from fastapi.params import Depends
from schemas.input import TextEmbedderRequest
from schemas.output import TextEmbedderResponse

from core.dependences import get_model
from rag_embedder.models.gemma import GemmaEmbedderDeployment

router = APIRouter()

@router.post("/google-embeddinggemma-300m")
async def generate_embeddings(
    request: TextEmbedderRequest, model: GemmaEmbedderDeployment = Depends(get_model)
) -> TextEmbedderResponse:
    """Генерирует эмбеддинг для текста.

    Args:
        request: Запрос с текстом для обработки.
        model: Модель для генерации эмбеддинга.

    Returns:
        Векторное представление текста (эмбеддинг).

    """
    return await model.embed(request)
