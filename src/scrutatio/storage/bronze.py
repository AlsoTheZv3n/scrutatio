"""Bronze layer: raw ClinicalTrials.gov studies, landed idempotently.

Bronze keeps the full API payload verbatim so Silver can be rebuilt without
re-fetching — the registry is a moving target and a re-fetch is not a rerun.

The write path is a single upsert on ``nct_id``. Running the same batch twice
updates in place rather than duplicating, which is what makes an interrupted
backfill safe to resume.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Final

from scrutatio.clients.ctgov import Study, last_update_posted, nct_id

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


@dataclass(frozen=True)
class TrialMeta:
    """The header a match result shows above the criterion breakdown."""

    nct_id: str
    title: str
    phase: str | None
    overall_status: str
    locations: list[str]


# One trial in the corpus carries 1,259 locations. Sending them all would make
# the payload the largest thing in the response and the list unreadable, so the
# tail is summarised rather than silently dropped — CT.gov has the full list and
# the UI links to it.
MAX_LOCATIONS: Final = 10


def _format_locations(raw_locations: str | None) -> list[str]:
    if not raw_locations:
        return []
    try:
        parsed = json.loads(raw_locations)
    except json.JSONDecodeError:  # pragma: no cover - Bronze holds API output
        return []
    if not isinstance(parsed, list):  # pragma: no cover
        return []

    shown = [
        ", ".join(
            part for part in (loc.get("facility"), loc.get("city"), loc.get("country")) if part
        )
        for loc in parsed[:MAX_LOCATIONS]
        if isinstance(loc, dict)
    ]
    if len(parsed) > MAX_LOCATIONS:
        shown.append(f"+{len(parsed) - MAX_LOCATIONS} more")
    return shown


def trial_metadata(db: duckdb.DuckDBPyConnection, nct_ids: Sequence[str]) -> dict[str, TrialMeta]:
    """Headers for the given trials, read out of the stored API payload.

    Bronze keeps the response verbatim precisely so this needs no second fetch:
    the registry is a moving target, and re-fetching would mean a match result
    could disagree with the criteria it was derived from.
    """
    if not nct_ids:
        return {}

    rows = db.execute(
        f"""
        SELECT nct_id,
               json_extract_string(raw, '$.protocolSection.identificationModule.briefTitle'),
               json_extract(raw, '$.protocolSection.designModule.phases')::VARCHAR,
               json_extract_string(raw, '$.protocolSection.statusModule.overallStatus'),
               json_extract(raw, '$.protocolSection.contactsLocationsModule.locations')::VARCHAR
        FROM {BRONZE_TABLE}
        WHERE nct_id IN (SELECT unnest(?::VARCHAR[]))
        """,  # noqa: S608 - table name is a module constant
        [list(nct_ids)],
    ).fetchall()

    out: dict[str, TrialMeta] = {}
    # Not `nct_id`: that name is the imported accessor from `scrutatio.clients.ctgov`, and
    # shadowing it here would work only for as long as nobody calls it.
    for identifier, title, phases_json, status, locations_json in rows:
        phases: list[str] = []
        if phases_json:
            try:
                loaded = json.loads(phases_json)
                phases = [str(p) for p in loaded] if isinstance(loaded, list) else []
            except json.JSONDecodeError:  # pragma: no cover
                phases = []
        out[str(identifier)] = TrialMeta(
            nct_id=str(identifier),
            title=str(title or ""),
            phase=", ".join(phases) or None,
            overall_status=str(status or "UNKNOWN"),
            locations=_format_locations(locations_json),
        )
    return out


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
