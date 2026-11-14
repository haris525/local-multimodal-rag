# Build context: repository root.
FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
COPY libs/common libs/common
COPY services/rag/pyproject.toml services/rag/uv.lock services/rag/
RUN cd services/rag && uv sync --frozen --no-dev --no-install-project --no-editable
COPY services/rag services/rag
RUN cd services/rag && uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 10001 app
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 HOME=/tmp \
    LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false
USER app
WORKDIR /app
EXPOSE 8002
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8002", "--no-access-log"]
