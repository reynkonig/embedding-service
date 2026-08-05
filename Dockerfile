# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.10.5 AS uv
FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/cache/huggingface \
    EMBEDDING_HOST=0.0.0.0 \
    EMBEDDING_PORT=8091

COPY --from=uv /uv /uvx /usr/local/bin/

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home app \
    && mkdir -p /app /cache/huggingface \
    && chown -R app:app /app /cache/huggingface

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY --chown=app:app embedding_service ./embedding_service

USER app
EXPOSE 8091
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8091/healthz', timeout=3).read()"]

CMD ["/app/.venv/bin/python", "-m", "embedding_service"]
