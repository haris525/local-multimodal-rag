# Build context: repository root.
FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
COPY libs/common libs/common
COPY services/auth/pyproject.toml services/auth/uv.lock services/auth/
RUN cd services/auth && uv sync --frozen --no-dev --no-install-project --no-editable
COPY services/auth services/auth
RUN cd services/auth && uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown app:app /data
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1
USER app
WORKDIR /app
EXPOSE 8007
CMD ["uvicorn", "auth.api:app", "--host", "0.0.0.0", "--port", "8007", "--no-access-log"]
