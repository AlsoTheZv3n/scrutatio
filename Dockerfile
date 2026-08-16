# Multi-stage: the builder carries uv and the build toolchain, the runtime carries
# neither. The pipeline needs no services — DuckDB is a file — so the only runtime
# dependencies are the Python environment and a writable volume.

# --- builder ---------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies resolve from the lockfile in their own layer, so editing source
# does not re-resolve them. --frozen fails rather than silently updating the lock.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --- runtime ---------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# Non-root. The image writes only to /data, which is a mounted volume.
RUN groupadd --system --gid 1001 scrutatio \
 && useradd --system --uid 1001 --gid scrutatio --create-home scrutatio \
 && mkdir -p /data \
 && chown -R scrutatio:scrutatio /data

COPY --from=builder --chown=scrutatio:scrutatio /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # The database lives on the volume, not in the layer. A container that loses
    # its volume loses the corpus, which a backfill rebuilds in ~7 minutes.
    DB_PATH=/data/scrutatio.duckdb

WORKDIR /app
USER scrutatio
VOLUME ["/data"]

# Exercises the real path: settings load, database opens, tables exist. Cheap and
# idempotent — `status` creates the schema if it is missing and reads nothing else.
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
    CMD ["scrutatio", "status"]

ENTRYPOINT ["scrutatio"]
CMD ["status"]
