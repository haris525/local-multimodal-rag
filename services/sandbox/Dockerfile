# Build context: repository root. Runs model-written Python: read-only root filesystem, tmpfs scratch, no network
# but rag's, no secrets, no volumes (compose.yaml, services/sandbox/src/sandbox/api.py).
FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
COPY libs/common libs/common
COPY services/sandbox/pyproject.toml services/sandbox/uv.lock services/sandbox/
RUN cd services/sandbox && uv sync --frozen --no-dev --no-install-project --no-editable
COPY services/sandbox services/sandbox
RUN cd services/sandbox && uv sync --frozen --no-dev --no-editable
# matplotlib's font cache, built once here and copied into every run (each run would otherwise spend seconds on it)
RUN MPLCONFIGDIR=/opt/matplotlib MPLBACKEND=Agg /app/.venv/bin/python -c "import matplotlib.font_manager"

# The tests on Linux as an unprivileged user, the RLIMIT and process-group ones included (skipped on Windows):
#   docker build --target test -f services/sandbox/Dockerfile .
FROM build AS test
RUN cd services/sandbox && uv sync --frozen --no-editable \
    && useradd --uid 10101 --no-create-home --shell /usr/sbin/nologin sandbox
USER sandbox
RUN cd services/sandbox && /app/.venv/bin/python -m pytest -q -p no:cacheprovider tests

FROM python:3.12-slim-bookworm
# Its own user id, shared with no other service: RLIMIT_NPROC counts every process of a user id on the host.
RUN useradd --uid 10101 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin sandbox \
    && find / -xdev -type f -perm /6000 -exec chmod a-s {} + \
    && mkdir -m 1777 /scratch
COPY --from=build /app/.venv /app/.venv
COPY --from=build /opt/matplotlib /opt/matplotlib
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HOME=/nonexistent \
    SANDBOX_SCRATCH_DIR=/scratch SANDBOX_WIPE_DIRS=/tmp,/dev/shm SANDBOX_SWEEP=1 SANDBOX_MPL_CACHE=/opt/matplotlib
USER sandbox
WORKDIR /app
EXPOSE 8008
CMD ["uvicorn", "sandbox.api:app", "--host", "0.0.0.0", "--port", "8008", "--no-access-log"]
