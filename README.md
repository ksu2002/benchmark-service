Монорепозиторий с двумя сервисами: платформа бенчмарков диалогов и HTTP-сервис эмбеддингов.

## Состав

| Каталог | Назначение |
|---------|------------|
| [`benchmark_service/`](benchmark_service/) | Streamlit-приложение: разметка JSONL, генерация диалогов, бенчмарки ассистента, LLM-судья, аналитика и фоновые прогоны (Postgres + MinIO + RabbitMQ) |
| [`embedder_gemma/`](embedder_gemma/) | FastAPI-сервис эмбеддингов на базе [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) |


## Быстрый старт

### Benchmark Service

```bash
cd benchmark_service
cp .env.template .env
docker compose up -d --build
```

UI: [http://localhost:8504](http://localhost:8504)

Подробности: [`benchmark_service/README.md`](benchmark_service/README.md)

### Embedder Gemma

```bash
cd embedder_gemma
cp .env.templates .env
docker compose -f docker-compose.dev.yml up --build
```

Подробности: [`embedder_gemma/README.md`](embedder_gemma/README.md)

## Ограничения публичной версии

В `benchmark_service` интеграции с **Dialog Manager** и **ClickHouse** заменены NDA-заглушками: загрузка логов из внутренней БД и прогон сценариев через DM в этой копии репозитория недоступны. Остаются рабочими разметка/бенчмарк по JSONL, LiteLLM, внешний URL ассистента, LLM-судья и инфраструктура фоновых запусков.

## Структура репозитория

```text
itmo/
├── benchmark_service/     # Streamlit + worker + tests
│   ├── config/            # runtime-конфиги (RabbitMQ)
│   ├── src/
│   └── tests/
├── embedder_gemma/        # FastAPI embedder
│   ├── src/rag_embedder/
│   └── tests/
└── README.md              # этот файл
```
