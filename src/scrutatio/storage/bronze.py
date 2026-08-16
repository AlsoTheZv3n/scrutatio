"""Bronze layer: raw ClinicalTrials.gov studies, landed idempotently.

Bronze keeps the full API payload verbatim so Silver can be rebuilt without
re-fetching — the registry is a moving target and a re-fetch is not a rerun.

The write path is a single upsert on ``nct_id``. Running the same batch twice
updates in place rather than duplicating, which is what makes an interrupted
backfill safe to resume.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import date
from typing import TYPE_CHECKING, Final

from scrutatio.ctgov import Study, last_update_posted, nct_id

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

BRONZE_TABLE: Final = "bronze_studies"
RUNS_TABLE: Final = "bronze_runs"

# Chunked so an interrupted backfill loses one batch, not the whole run.
DEFAULT_BATCH_SIZE: Final = 500


def ensure_storage(db: duckdb.DuckDBPyConnection) -> None:
    """Create the Bronze tables if they do not exist."""
    db.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {BRONZE_TABLE} (
            nct_id VARCHAR PRIMARY KEY,
            last_update_posted DATE,
            raw VARCHAR NOT NULL,
            ingested_at TIMESTAMP NOT NULL
        )
        """
    )
    # The incremental watermark lives here, not in max(bronze.last_update_posted).
    # Deriving it from Bronze is unsafe: a run that dies at batch 12 of 23 still
    # leaves the newest studies it did land, so the next incremental pass starts
    # after them and never fetches the ones that never arrived.
    db.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {RUNS_TABLE} (
            run_id VARCHAR NOT NULL,
            mode VARCHAR NOT NULL,
            started_at TIMESTAMP NOT NULL,
            finished_at TIMESTAMP,
            studies_written BIGINT,
            watermark DATE,
            complete BOOLEAN NOT NULL
        )
        """
    )


def _rows(studies: Iterable[Study]) -> list[tuple[str, date | None, str]]:
    """Flatten studies to insertable rows, deduplicated on ``nct_id``.

    The deduplication is not defensive padding: CT.gov paginates over a live
    index, so a study updated mid-walk can surface on two consecutive pages.
    ``INSERT … ON CONFLICT DO UPDATE`` refuses to touch the same row twice within
    one statement, so a repeated key aborts the batch. Last occurrence wins.
    """
    records: dict[str, tuple[str, date | None, str]] = {}
    skipped = 0
    for study in studies:
        identifier = nct_id(study)
        if identifier is None:
            skipped += 1
            continue
        records[identifier] = (
            identifier,
            last_update_posted(study),
            study.model_dump_json(by_alias=True, exclude_none=True),
        )

    if skipped:
        logger.warning("Skipped %d studies with no NCT id", skipped)
    return list(records.values())


def write_bronze(
    db: duckdb.DuckDBPyConnection, studies: Iterable[Study], *, batch: str = "-"
) -> int:
    """Land one batch of studies into Bronze. Returns the row count written.

    ``batch`` is log context only; it no longer names a staging file.
    """
    rows = _rows(studies)
    if not rows:
        return 0

    # Values are bound, never interpolated — trial text is arbitrary prose.
    # Dates arrive as typed `date` objects straight from the generated model, so
    # the `try_cast` the old COPY INTO path needed is gone with it.
    db.executemany(
        f"""
        INSERT INTO {BRONZE_TABLE} (nct_id, last_update_posted, raw, ingested_at)
        VALUES (?, ?, ?, current_timestamp)
        ON CONFLICT (nct_id) DO UPDATE SET
            last_update_posted = excluded.last_update_posted,
            raw = excluded.raw,
            ingested_at = excluded.ingested_at
        """,  # noqa: S608 - table name is a module constant
        rows,
    )

    logger.info("Landed %d studies into Bronze from batch %s", len(rows), batch)
    return len(rows)


def bronze_count(db: duckdb.DuckDBPyConnection) -> int:
    """Number of studies currently in Bronze."""
    row = db.execute(f"SELECT count(*) FROM {BRONZE_TABLE}").fetchone()  # noqa: S608
    return int(row[0]) if row else 0


def bronze_max_update(db: duckdb.DuckDBPyConnection) -> date | None:
    """Newest ``last_update_posted`` present in Bronze.

    Diagnostic only. Do NOT use this as the incremental starting point — see
    ``safe_watermark``.
    """
    row = db.execute(f"SELECT max(last_update_posted) FROM {BRONZE_TABLE}").fetchone()  # noqa: S608
    return row[0] if row and row[0] is not None else None


def safe_watermark(db: duckdb.DuckDBPyConnection) -> date | None:
    """Watermark from the last run that finished completely.

    Returns ``None`` when no complete run exists, which correctly forces a full
    pass rather than resuming from a point that was never actually reached.
    """
    row = db.execute(
        f"""
        SELECT watermark FROM {RUNS_TABLE}
        WHERE complete AND watermark IS NOT NULL
        ORDER BY finished_at DESC
        LIMIT 1
        """  # noqa: S608
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def start_run(db: duckdb.DuckDBPyConnection, run_id: str, mode: str) -> None:
    """Record a run as started and incomplete."""
    db.execute(
        f"""
        INSERT INTO {RUNS_TABLE}
        (run_id, mode, started_at, finished_at, studies_written, watermark, complete)
        VALUES (?, ?, current_timestamp, NULL, NULL, NULL, false)
        """,  # noqa: S608
        [run_id, mode],
    )


def finish_run(db: duckdb.DuckDBPyConnection, run_id: str, *, written: int, complete: bool) -> None:
    """Mark a run finished.

    Only a ``complete`` run publishes a watermark; a partial or limited run
    leaves it NULL so the next incremental pass ignores it entirely.
    """
    watermark = (
        f"(SELECT max(last_update_posted) FROM {BRONZE_TABLE})"  # noqa: S608 - constant table
        if complete
        else "NULL"
    )
    db.execute(
        f"""
        UPDATE {RUNS_TABLE} SET
            finished_at = current_timestamp,
            studies_written = ?,
            watermark = {watermark},
            complete = ?
        WHERE run_id = ?
        """,  # noqa: S608
        [int(written), complete, run_id],
    )
