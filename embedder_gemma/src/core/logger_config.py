"""Настройки работы сервиса."""

from core.config import settings

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelname)s \t %(asctime)s - %(name)s - %(message)s",
            "datefmt": "%d/%m/%Y %H:%M:%S",
        }
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        }
    },
    "loggers": {
        "": {"handlers": ["default"], "level": settings.log_level.upper()},
        "uvicorn.error": {"level": settings.log_level.upper()},
    },
}
