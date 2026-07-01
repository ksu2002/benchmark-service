"""Нагрузочное тестирование."""

from pathlib import Path
from random import choice

from locust import HttpUser, task


class LocustTest(HttpUser):
    """Класс для нагрузочного тестирования эндпоинта генерации эмбеддингов."""

    @task
    def embedder(self):
        """Отправляет POST-запрос для генерации эмбеддинга случайного текста.

        Raises:
            Exception: Если произошла ошибка при отправке запроса.

        """
        self.client.post(
            "/dialog/nlp/embedding/google-embeddinggemma-300m",
            json={"text": choice(self._texts)},
        )

    @staticmethod
    def get_test_data(path) -> list[str]:
        """Возвращает данные для нагрузочного тестирования.

        Args:
            path: Путь до файла с предложениями.

        Returns:
            Тестовые предложения.

        """
        with Path(path).open() as file:
            return file.readlines()

    _texts = get_test_data("tests/stress/data/test_samples.txt")
