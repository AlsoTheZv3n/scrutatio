"""Bronze layer: raw ClinicalTrials.gov studies, landed idempotently.

Bronze keeps the full API payload verbatim so Silver can be rebuilt without
re-fetching — the registry is a moving target and a re-fetch is not a rerun.

The write path is upload-then-``MERGE``: studies are serialised to newline
JSON, uploaded to a Unity Catalog volume, loaded into a staging table with
``COPY INTO``, then merged on ``nct_id``. Running the same batch twice updates
in place rather than duplicating, which is what makes an interrupted backfill
safe to resume.
"""

from __future__ import annotations

import gzip
import json
import logging
import re
from collections.abc import Iterable
from typing import Final

from scrutatio.config import get_settings
from scrutatio.ctgov import Study, last_update_posted, nct_id
from scrutatio.storage.sql import SqlClient

logger = logging.getLogger(__name__)

# Resolved once at import from settings. The catalog name is workspace-specific:
# Databricks Free Edition on AWS ships a catalog called "workspace", while an
# Azure serverless workspace names it after the workspace itself. Hardcoding it
# meant the first statement against the new workspace failed on an unknown
# catalog rather than doing the work.
CATALOG: Final = get_settings().catalog
SCHEMA: Final = get_settings().schema_name
VOLUME: Final = "landing"
BRONZE_TABLE: Final = f"{CATALOG}.{SCHEMA}.bronze_studies"
RUNS_TABLE: Final = f"{CATALOG}.{SCHEMA}.bronze_runs"
_STAGING_TABLE: Final = f"{CATALOG}.{SCHEMA}._bronze_staging"

# Chunked so an interrupted backfill loses one batch, not the whole run.
DEFAULT_BATCH_SIZE: Final = 500

# The batch name is interpolated into a path literal inside COPY INTO, which
# the Statement Execution API cannot parameterise. A quote in the name would
# break out of the literal, so the charset is restricted instead.
_SAFE_BATCH_NAME: Final = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class UnsafeBatchNameError(ValueError):
    """The batch name contains characters that cannot be safely interpolated."""


def _validate_batch(batch: str) -> str:
    if not _SAFE_BATCH_NAME.match(batch):
        msg = (
            f"Batch name {batch!r} must match {_SAFE_BATCH_NAME.pattern} — it is interpolated "
            f"into a SQL path literal and cannot be parameterised"
        )
        raise UnsafeBatchNameError(msg)
    return batch


def ensure_storage(sql: SqlClient) -> None:
    """Create the schema, volume and Bronze table if they do not exist."""
    sql.execute(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    sql.execute(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
    sql.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {BRONZE_TABLE} (
            nct_id STRING NOT NULL,
            last_update_posted DATE,
            raw STRING NOT NULL,
            ingested_at TIMESTAMP NOT NULL
        )
        USING DELTA
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """
    )
    # The incremental watermark lives here, not in max(bronze.last_update_posted).
    # Deriving it from Bronze is unsafe: a run that dies at batch 12 of 23 still
    # leaves the newest studies it did land, so the next incremental pass starts
    # after them and never fetches the ones that never arrived.
    sql.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {RUNS_TABLE} (
            run_id STRING NOT NULL,
            mode STRING NOT NULL,
            started_at TIMESTAMP NOT NULL,
            finished_at TIMESTAMP,
            studies_written BIGINT,
            watermark DATE,
            complete BOOLEAN NOT NULL
        )
        USING DELTA
        """
    )


def _to_ndjson(studies: Iterable[Study]) -> tuple[bytes, int]:
    """Serialise studies to gzipped newline JSON. Returns (payload, row count).

    Deduplicates on ``nct_id``, last occurrence winning. This is not defensive
    padding: CT.gov paginates over a live index, so a study updated mid-walk can
    surface on two consecutive pages. Delta's MERGE raises
    ``DELTA_MULTIPLE_SOURCE_ROW_MATCHING_TARGET_ROW`` when the source repeats a
    key that already exists in the target, and silently inserts both when it does
    not — an aborted backfill or a duplicated row, neither acceptable.
    """
    records: dict[str, str] = {}
    skipped = 0
    for study in studies:
        identifier = nct_id(study)
        if identifier is None:
            skipped += 1
            continue
        posted = last_update_posted(study)
        records[identifier] = json.dumps(
            {
                "nct_id": identifier,
                "last_update_posted": posted.isoformat() if posted else None,
                "raw": study.model_dump_json(by_alias=True, exclude_none=True),
            },
            ensure_ascii=False,
        )

    if skipped:
        logger.warning("Skipped %d studies with no NCT id", skipped)

    payload = gzip.compress(("\n".join(records.values()) + "\n").encode("utf-8"))
    return payload, len(records)


def write_bronze(sql: SqlClient, studies: Iterable[Study], *, batch: str) -> int:
    """Land one batch of studies into Bronze. Returns the row count written.

    ``batch`` names the staging file; reusing a name overwrites it, so a retried
    batch does not accumulate orphaned files in the volume.
    """
    _validate_batch(batch)

    payload, rows = _to_ndjson(studies)
    if rows == 0:
        return 0

    volume_path = f"{CATALOG}/{SCHEMA}/{VOLUME}/{batch}.ndjson.gz"
    sql.upload_file(volume_path, payload)

    sql.execute(f"DROP TABLE IF EXISTS {_STAGING_TABLE}")
    # Every staging column is STRING: COPY INTO infers JSON scalars as strings,
    # and a DATE column here fails with DELTA_FAILED_TO_MERGE_FIELDS. Casting
    # happens in the MERGE, where a malformed date becomes NULL instead of
    # aborting the whole batch.
    sql.execute(
        f"""
        CREATE TABLE {_STAGING_TABLE} (
            nct_id STRING,
            last_update_posted STRING,
            raw STRING
        ) USING DELTA
        """
    )
    sql.execute(
        f"""
        COPY INTO {_STAGING_TABLE}
        FROM '/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{batch}.ndjson.gz'
        FILEFORMAT = JSON
        FORMAT_OPTIONS ('multiLine' = 'false', 'compression' = 'gzip')
        COPY_OPTIONS ('force' = 'true')
        """
    )
    # MERGE, not INSERT: the same NCT id reappears on every incremental run.
    # S608: the only interpolated values are module-level table constants; every
    # value from the API travels through the uploaded file, never the statement.
    sql.execute(
        f"""
        MERGE INTO {BRONZE_TABLE} AS target
        USING {_STAGING_TABLE} AS source
        ON target.nct_id = source.nct_id
        WHEN MATCHED THEN UPDATE SET
            target.last_update_posted = try_cast(source.last_update_posted AS DATE),
            target.raw = source.raw,
            target.ingested_at = current_timestamp()
        WHEN NOT MATCHED THEN INSERT
            (nct_id, last_update_posted, raw, ingested_at)
        VALUES
            (
                source.nct_id,
                try_cast(source.last_update_posted AS DATE),
                source.raw,
                current_timestamp()
            )
        """  # noqa: S608
    )
    sql.execute(f"DROP TABLE IF EXISTS {_STAGING_TABLE}")

    logger.info("Landed %d studies into Bronze from batch %s", rows, batch)
    return rows


def bronze_count(sql: SqlClient) -> int:
    """Number of studies currently in Bronze."""
    rows = sql.execute(f"SELECT count(*) FROM {BRONZE_TABLE}")  # noqa: S608 - constant table name
    return int(rows[0][0]) if rows else 0


def bronze_max_update(sql: SqlClient) -> str | None:
    """Newest ``last_update_posted`` present in Bronze.

    Diagnostic only. Do NOT use this as the incremental starting point — see
    ``safe_watermark``.
    """
    rows = sql.execute(f"SELECT max(last_update_posted) FROM {BRONZE_TABLE}")  # noqa: S608
    return rows[0][0] if rows and rows[0][0] is not None else None


def safe_watermark(sql: SqlClient) -> str | None:
    """Watermark from the last run that finished completely.

    Returns ``None`` when no complete run exists, which correctly forces a full
    pass rather than resuming from a point that was never actually reached.
    """
    rows = sql.execute(
        f"""
        SELECT watermark FROM {RUNS_TABLE}
        WHERE complete AND watermark IS NOT NULL
        ORDER BY finished_at DESC
        LIMIT 1
        """  # noqa: S608
    )
    return rows[0][0] if rows and rows[0][0] is not None else None


def start_run(sql: SqlClient, run_id: str, mode: str) -> None:
    """Record a run as started and incomplete."""
    _validate_batch(run_id)
    sql.execute(
        f"""
        INSERT INTO {RUNS_TABLE}
        (run_id, mode, started_at, finished_at, studies_written, watermark, complete)
        VALUES ('{run_id}', '{mode}', current_timestamp(), NULL, NULL, NULL, false)
        """  # noqa: S608
    )


def finish_run(sql: SqlClient, run_id: str, *, written: int, complete: bool) -> None:
    """Mark a run finished.

    Only a ``complete`` run publishes a watermark; a partial or limited run
    leaves it NULL so the next incremental pass ignores it entirely.
    """
    _validate_batch(run_id)
    watermark = (
        f"(SELECT max(last_update_posted) FROM {BRONZE_TABLE})"  # noqa: S608 - constant table
        if complete
        else "NULL"
    )
    sql.execute(
        f"""
        UPDATE {RUNS_TABLE} SET
            finished_at = current_timestamp(),
            studies_written = {int(written)},
            watermark = {watermark},
            complete = {"true" if complete else "false"}
        WHERE run_id = '{run_id}'
        """  # noqa: S608
    )
