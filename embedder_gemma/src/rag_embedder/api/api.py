"""Маршруты API."""

from api.endpoints import embed, health
from fastapi import APIRouter

router = APIRouter()

router.include_router(embed.router, prefix="/dialog/nlp/embedding", tags=["Embedding"])
router.include_router(health.router, prefix="/health", tags=["Health"])
