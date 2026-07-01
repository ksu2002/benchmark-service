"""Эндпоинты для проверки работоспособности сервиса."""

from fastapi import APIRouter, Depends, HTTPException

from core.dependences import get_model

router = APIRouter()

@router.get("")
async def health_check(model = Depends(get_model)):
    """Проверяет работоспособность сервиса и состояние модели.

    Raises:
        HTTPException: Если сервис недоступен или модель не работает.

    Returns:
        Статус работоспособности сервиса.

    """
    status = model.check_health()
    if not status:
        raise HTTPException(status_code=503, detail="Service unavailable")
    return {"status": "OK"}
