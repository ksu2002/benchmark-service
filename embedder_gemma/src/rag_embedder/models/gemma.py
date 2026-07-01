"""Логика работы с моделью."""

import logging

from clearml import Model
from requests import RequestException
from schemas.input import TextEmbedderRequest
from schemas.output import TextEmbedderResponse
from sentence_transformers import SentenceTransformer
from torch import cuda, device, no_grad

from core.config import settings

_logger = logging.getLogger(__name__)

class GemmaEmbedderDeployment:
    """Класс модели."""

    def __init__(self):
        """Инициализирует модель для генерации эмбеддингов."""
        try:
            _logger.info("Loading model from ClearML...")
            self.clearml_model = Model(model_id=settings.clearml_model_id)
            self.weights_path = self.clearml_model.get_local_copy()
            _logger.info("Model is downloaded from ClearML")
        except ValueError as e:
            _logger.error(
                "Invalid model ID or model metadata error for id=%r: %s",
                settings.clearml_model_id,
                e,
            )
            raise
        except RequestException as e:
            _logger.error(
                "Network/HTTP error while accessing ClearML server for model id=%r: %s",
                settings.clearml_model_id,
                e,
            )
            raise
        except OSError as e:
            _logger.error(
                "Local filesystem error while downloading model weights for id=%r: %s",
                settings.clearml_model_id,
                e,
            )
            raise
        except RuntimeError as e:
            _logger.error(
                "Runtime error while working with ClearML model id=%r: %s",
                settings.clearml_model_id,
                e,
            )
            raise
        except Exception as e:
            _logger.exception(
                "Unexpected error while loading model from ClearML (id=%r): %s",
                settings.clearml_model_id,
                e,
            )
            raise

        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.model = SentenceTransformer(self.weights_path).to(self.device)

        if cuda.is_available():
            _logger.info("Model is loaded to GPU")
        else:
            _logger.info("Model is loaded to CPU")


    def check_health(self) -> bool:
        """Проверяет работоспособность модели."""
        try:
            sentence = "This is an example sentence"
            with no_grad():
                self.model.encode_query(sentence)
            return True

        except Exception:
            return False


    async def embed(self, request: TextEmbedderRequest) -> TextEmbedderResponse:
        """Генерирует эмбеддинг для текста.

        Args:
            request: Запрос с текстом для обработки.

        Returns:
            Эмбеддинг текста в виде списка чисел.

        """
        _logger.info(f"Request: {request.text[: settings.max_request_value_length]}")

        with no_grad():
            embedding = self.model.encode_query(request.text)

        return TextEmbedderResponse(embedding=embedding.squeeze().tolist())
