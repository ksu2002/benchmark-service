# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

ARG PIP_INDEX_URL=
ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8502 \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1024

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -U pip \
    && if [ -n "$PIP_INDEX_URL" ]; then \
         python -m pip config set global.index-url "$PIP_INDEX_URL"; \
       fi \
    && python -m pip install -r requirements.txt

COPY . .

RUN useradd --create-home --uid 1000 --user-group appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8502

CMD ["streamlit", "run", "/app/src/home.py"]
