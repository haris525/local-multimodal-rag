# Build context: repository root.
FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
COPY libs/common libs/common
COPY services/ingest/pyproject.toml services/ingest/uv.lock services/ingest/
RUN cd services/ingest && uv sync --frozen --no-dev --no-install-project --no-editable
COPY services/ingest services/ingest
RUN cd services/ingest && uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /data && chown app:app /data
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 HF_HOME=/tmp/hf \
    OMP_NUM_THREADS=8 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
# Docling layout, table and OCR weights are baked in, then verified with the network disabled.
RUN docling-tools models download layout tableformer rapidocr -o /opt/docling-models \
    && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m ingest.warmup /opt/docling-models \
    && chmod -R a+rX /opt/docling-models && rm -rf /tmp/hf
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
USER app
WORKDIR /app
EXPOSE 8001
CMD ["uvicorn", "ingest.api:app", "--host", "0.0.0.0", "--port", "8001", "--no-access-log"]
