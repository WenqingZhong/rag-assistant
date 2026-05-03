FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

WORKDIR /app

# Copy dependency files first (Docker caches this layer — faster rebuilds)
COPY pyproject.toml uv.lock ./

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Copy source code
COPY src /app/src

FROM python:3.12.8-slim AS final

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY --from=base /app /app
ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]