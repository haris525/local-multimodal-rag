# Build context: repository root.
FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
COPY libs/common libs/common
COPY eval/pyproject.toml eval/uv.lock eval/
RUN cd eval && uv sync --frozen --no-dev --no-install-project --no-editable
COPY eval eval
RUN cd eval && uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 10001 app && mkdir -p /eval/datasets /eval/reports && chown -R app:app /eval
COPY --from=build /app/.venv /app/.venv
COPY eval/benchmarks /eval/benchmarks
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
USER app
WORKDIR /eval
ENTRYPOINT ["rag-eval"]
