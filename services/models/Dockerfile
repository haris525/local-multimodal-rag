# Build context: repository root. CUDA runtime libraries ship inside the PyTorch wheels;
# the NVIDIA driver comes from the host through the container toolkit.
FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
COPY libs/common libs/common
COPY services/models/pyproject.toml services/models/uv.lock services/models/
RUN cd services/models && uv sync --frozen --no-dev --no-install-project --no-editable
COPY services/models services/models
RUN cd services/models && uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 10001 app
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 HF_HOME=/tmp/hf \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
USER app
WORKDIR /app
EXPOSE 8003
CMD ["uvicorn", "models_svc.api:app", "--host", "0.0.0.0", "--port", "8003", "--no-access-log"]
