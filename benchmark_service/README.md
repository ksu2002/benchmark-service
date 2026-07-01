# Benchmark Service

Streamlit-приложение для подготовки, разметки, генерации, анализа и оценки диалогов между пользователем и ассистентом.

## Что делает проект

- размечает и курирует JSONL-датасеты диалогов
- генерирует новые диалоги через LiteLLM
- запускает бенчмарки ассистента (LiteLLM, внешний URL; DM — NDA)
- хранит фоновые запуски в Postgres, MinIO и RabbitMQ
- калибрует LLM-судью
- анализирует результаты, ошибки и кластеры

## Зачем нужна `config/`

В `config/` лежит только runtime-конфиг инфраструктуры:

- `config/rabbitmq/advanced.config` — монтируется в контейнер RabbitMQ (`docker-compose.yml`).

Шаблоны бенчмарка в JSON **не хранятся в репозитории**: конфиг прогона собирается в UI страниц бенчмарка или сохраняется в Postgres при постановке задачи в очередь.

## Ограничения публичной версии

Интеграции с Dialog Manager и ClickHouse заменены NDA-заглушками. Работают разметка по JSONL, бенчмарк через LiteLLM / внешний URL, LLM-судья, кластеризация и фоновые прогоны.

## Структура `src/`

Код разложен по доменным пакетам:

```text
src/
├── analysis/       # аналитика выборок и ошибок
├── benchmarking/   # ядро бенчмарка, воркер, parsing, config_models
├── clustering/     # кластеризация и текстовые утилиты
├── common/         # общие helper'ы
├── integrations/   # внешние интеграции
├── judge/          # логика LLM-судьи
├── pages/          # Streamlit-страницы
├── storage/        # Postgres / MinIO / очередь / миграции схемы
├── tools/          # вспомогательные CLI/IO-инструменты
├── ui/             # UI-компоненты и вспомогательные экраны
├── benchmark_runner.py
├── dialog_clustering.py
└── home.py
```


```bash
cp .env.template .env
docker compose up -d --build
```

UI будет доступен на `http://localhost:8504`.

Порты:

- `8504` — Streamlit
- `5434` — Postgres
- `9004` — MinIO API
- `9005` — MinIO Console
- `5674` — RabbitMQ AMQP
- `15674` — RabbitMQ UI

Если PyPI медленный, можно задать в `.env`:

```env
PIP_INDEX_URL=https://mirror.yandex.ru/mirrors/pypi/simple
```

## Локальный запуск

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt

cp .env.template .env

# UI
set PYTHONPATH=src
streamlit run src/home.py --server.port=8502

# Воркер
set PYTHONPATH=src
python -m benchmarking.worker
```

Для Linux/macOS вместо `set` используйте `export`.

## Docker и compose

Что изменено в контейнерной части:

- `Dockerfile` задаёт `PYTHONPATH=/app/src` внутри образа
- Streamlit стартует через `src/home.py`, что важно для multipage-режима
- воркер запускается как `python -m benchmarking.worker`
- сетевые и Streamlit-настройки вынесены в `ENV`, чтобы образ был предсказуемее

## Страницы приложения

- `pages/01_Разметка.py` — первичная разметка
- `pages/02_Генерация.py` — генерация диалогов
- `pages/03_Анализ_и_редактирование_данных.py` — анализ и курирование датасета
- `pages/04_Бенчмарк.py` — бенчмарк ассистента
- `pages/05_Разметка_из_сценария.py` — разметка из логов сценариев (ClickHouse — NDA)
- `pages/06_Бенчмарк_сценария.py` — бенчмарк Dialog Manager (NDA)
- `pages/07_Результаты.py` — результаты запусков
- `pages/08_Настройки_LLM_судьи.py` — калибровка LLM-судьи
- `pages/09_Бенчмарк_агентов.py` — анализ агентов по Langfuse

## Полезные команды

```bash
# Пересобрать UI и воркер
docker compose up -d --build streamlit-dev worker-dev-1

# Логи воркера
docker compose logs -f worker-dev-1

# Прогон тестов
python -m unittest discover -s tests -p "test_*.py"
```

## Тесты

Сейчас добавлены базовые unit-тесты на:

- парсинг benchmark-кейсов
- безопасный custom_eval
- JSONL-утилиты
- аналитику выборок
- чистые текстовые функции кластеризации

Запуск:

```bash
python -m unittest discover -s tests -p "test_*.py"
```
