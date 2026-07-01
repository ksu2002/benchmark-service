"""Модуль конфигурации приложения."""


from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения."""

    debug: bool = Field(
        default=False,
        description="Включает режим отладки. При значении True автоматически устанавливает"
        "log_level в 'debug'.",
    )
    log_level: str = Field(
        default="info",
        description="Уровень логирования (например, 'info', 'debug', 'warning')."
        "Может быть переопределён через debug.",
    )
    max_request_value_length: int = Field(
        default=200, description="Максимальная длина запроса для вывода в логах."
    )
    clearml_model_id: str = Field(description="ID модели из ClearML.")
    model_config = SettingsConfigDict(
        env_file= ".env",
        extra="ignore",
    )

    @model_validator(mode="after")
    def handle_debug_log_level(self) -> "Settings":
        """Обрабатывает логику DEBUG и LOG_LEVEL."""
        if self.debug:
            self.log_level = "debug"
        return self


settings = Settings()
