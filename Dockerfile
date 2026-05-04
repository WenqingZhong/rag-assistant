FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

WORKDIR /app

COPY pyproject.toml uv.lock ./

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY src /app/src

FROM python:3.12.8-slim AS final

RUN apt-get update && apt-get install -y \
    libxcb1 \
    libx11-6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY --from=base /app /app
ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]