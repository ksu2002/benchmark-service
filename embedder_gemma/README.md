# Embedder сервис

- Использует [HF модель](https://huggingface.co/google/embeddinggemma-300m)
- Название endpont'a `/dialog/nlp/embedding/google-embeddinggemma-300m` для совместимости c базой знаний в "Диалог"
## Где вызывается
- База знаний в платформе "Диалог"

##  Архитектура
- для создания API используется `fastapi`
```
├── src
│   └── rag_embedder
│       ├── api
│       │   ├── api.py
│       │   └── endpoints
│       │       ├── embed.py    # Создание эмбеддингов
│       │       └── health.py   # Проверка состояния сервиса
│       ├── config.py           # Загрузка переменных окружения для конфигурации приложения
│       ├── main.py             # Создание и настройка FastAPI приложения
│       ├── models
│       │   └── gemma.py        # Модель эмбеддера
│       └── schemas
│           ├── input.py        # Входные данные
│           └── output.py       # Выходные данные
├── tests
│   └── stress
│       ├── data
│       │   └── test_samples.txt
│       └── locust.py            # Нагрузочное тестирование
├── uv.lock
├── docker-compose.dev.yml
├── docker-compose.prod.yml
├── Dockerfile.dev
├── Dockerfile.prod
├── pyproject.toml
└── README.md
```

## Инструкция по поднятию сервиса

### Описание переменных окружения

|Переменная|Описание|
|-----|-----|
|`BACKEND_VERSION`|Версия бэкенда приложения, указываемая в формате тега|
|`DEBUG`|Режим отладки. Если установлено в `true`, приложение будет выводить дополнительную информацию.|
|`CLEARML_MODEL_ID`|ID модели в ClearML (например, `<clearml_model_id>`)|
|`NVIDIA_VISIBLE_DEVICES`|Указывает, какие устройства NVIDIA GPU должны быть видны контейнеру (для работы на gpu)|
|`NVIDIA_DRIVER_CAPABILITIES`|Определяет, какие возможности драйвера NVIDIA должны быть доступны внутри контейнера (для работы на gpu)|
|`PYTORCH_CUDA_ALLOC_CONF`|Настройка конфигурации выделения памяти CUDA для PyTorch (для работы на gpu)|
|`LOG_LEVEL`|Уровень логирования|
|`MAX_LOG_MESSAGE_LENGTH`|Максимальная длина лог-сообщения|
|`APP_PORT`|Порт, на котором приложение будет слушать входящие запросы|
### Запуск (локально)

- склонируйте проект

- по примеру в `.env.templates` создайте `.env` файл с переменными окружения и заполните их.
- запустите сборку сервиса:

```bash
docker compose -f docker-compose.dev.yml build
```

- для запуска сервиса:

```bash
docker compose -f docker-compose.dev.yml up -d
```

##  Тестирование

### Нагрузочное тестирование
- для запуска:
```bash
pip install locust==2.32.*
PYTHONPATH=tests/ locust -f tests/stress/locust.py --host=<хост-сервиса>