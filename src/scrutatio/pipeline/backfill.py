"""Bronze backfill: land the configured scope into DuckDB, in resumable batches.

Two modes:

* **Full** — walk the whole scope. Cheap on the CT.gov side (~19 requests for
  11,200 studies) and safe to rerun, because Bronze is upserted on ``nct_id``
  rather than appended.
* **Incremental** — start from the watermark of the last *completed* run and
  fetch only what changed since. This is the mode a schedule runs.

Batching exists for failure containment, not throughput: an interrupted run
loses at most one batch, and rerunning re-lands it without duplicating.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from itertools import islice
from typing import TYPE_CHECKING

from scrutatio.clients.ctgov import CtGovClient, Study
from scrutatio.storage import (
    DEFAULT_BATCH_SIZE,
    bronze_count,
    ensure_storage,
    finish_run,
    safe_watermark,
    start_run,
    write_bronze,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import duckdb

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackfillResult:
    """What a backfill run did."""

    written: int
    batches: int
    rows_before: int
    rows_after: int
    incremental_from: date | None
    complete: bool = False
    """True only if the full scope was walked. A partial or limited run does not
    publish a watermark, so the next incremental pass will not resume past it."""

    @property
    def added(self) -> int:
        """Net new studies. Lower than ``written`` when rows were updates."""
        return self.rows_after - self.rows_before


def _chunks(items: Iterable[Study], size: int) -> Iterator[list[Study]]:
    iterator = iter(items)
    while chunk := list(islice(iterator, size)):
        yield chunk


def run_backfill(
    *,
    db: duckdb.DuckDBPyConnection,
    ctgov: CtGovClient,
    incremental: bool = False,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    batch_prefix: str = "backfill",
    run_token: str,
) -> BackfillResult:
    """Land studies into Bronze and report what changed."""
    ensure_storage(db)
    rows_before = bronze_count(db)

    since: date | None = None
    if incremental:
        since = safe_watermark(db)
        if since is None:
            logger.info("No completed run to resume from — falling back to a full pass")
        else:
            logger.info("Incremental pass from %s", since)

    run_id = f"{batch_prefix}-{run_token}"
    start_run(db, run_id, "incremental" if since else "full")

    written = 0
    batches = 0
    completed = False
    try:
        studies = ctgov.iter_studies(updated_since=since, limit=limit)
        for index, chunk in enumerate(_chunks(studies, batch_size)):
            written += write_bronze(db, chunk, batch=f"{run_id}-{index:04d}")
            batches += 1
            logger.info("Batch %d done — %d studies landed so far", index, written)
        # A limited run walked only part of the scope, so its endpoint is not a
        # valid resume point even though nothing failed.
        completed = limit is None
    finally:
        finish_run(db, run_id, written=written, complete=completed)
        if not completed:
            logger.warning(
                "Run %s did not complete the scope — the watermark was not advanced", run_id
            )

    rows_after = bronze_count(db)
    return BackfillResult(
        written=written,
        batches=batches,
        rows_before=rows_before,
        rows_after=rows_after,
        incremental_from=since,
        complete=completed,
    )
