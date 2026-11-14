# Build context: repository root. Runs with a read-only root filesystem and a read-only data mount.
FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
COPY libs/common libs/common
COPY services/sql/pyproject.toml services/sql/uv.lock services/sql/
RUN cd services/sql && uv sync --frozen --no-dev --no-install-project --no-editable
COPY services/sql services/sql
RUN cd services/sql && uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 10001 app
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 HOME=/tmp
USER app
WORKDIR /app
EXPOSE 8005
CMD ["uvicorn", "sql_sandbox.api:app", "--host", "0.0.0.0", "--port", "8005", "--no-access-log"]
