"""Silver extraction run: Bronze prose in, typed predicates out.

The run is designed to be stopped and restarted. Each batch is committed before
the next is fetched, so an interrupted pass keeps everything it finished, and
the next start re-queries what is still pending rather than replaying from the
beginning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scrutatio.extraction import EligibilityExtractor, RunStats, extract_many
from scrutatio.storage.silver import (
    ensure_silver,
    extraction_signature,
    pending_trials,
    silver_stats,
    write_silver,
)

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

# Small enough that an interrupted run loses little, large enough that the
# per-batch SQL overhead stays a rounding error next to the LLM calls.
DEFAULT_BATCH_SIZE = 100


@dataclass(frozen=True)
class ExtractionRunResult:
    """What one extraction pass accomplished."""

    signature: str
    trials_written: int
    criteria_written: int
    succeeded: int
    failed: int
    rate_limited: int
    remaining: int

    @property
    def quota_exhausted(self) -> bool:
        """True when throttling, rather than the work running out, ended the pass."""
        return self.remaining > 0 and self.rate_limited > 0 and self.failed > 0


def run_extraction(
    *,
    db: duckdb.DuckDBPyConnection,
    extractor: EligibilityExtractor,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = 4,
    retry_failed: bool = False,
    run_token: str,
) -> ExtractionRunResult:
    """Extract pending trials into Silver, committing batch by batch."""
    ensure_silver(db)
    signature = extraction_signature()
    logger.info("Extraction signature %s", signature)

    stats = RunStats()
    trials_written = 0
    criteria_written = 0
    batch_index = 0

    while limit is None or trials_written < limit:
        remaining_budget = None if limit is None else limit - trials_written
        take = batch_size if remaining_budget is None else min(batch_size, remaining_budget)

        pending = pending_trials(db, signature, limit=take, retry_failed=retry_failed)
        if not pending:
            break

        outcomes = list(extract_many(extractor, pending, max_workers=workers, stats=stats))
        rows, trials = write_silver(
            db,
            outcomes,
            batch=f"silver-{run_token}-{batch_index:04d}",
            signature=signature,
        )
        criteria_written += rows
        trials_written += trials
        batch_index += 1
        logger.info(
            "Batch %d: %d trials, %d criteria (%d failed so far)",
            batch_index,
            trials,
            rows,
            stats.failed,
        )

        # Every trial in the batch failed and the endpoint was throttling —
        # continuing just burns the remaining attempts against a closed door.
        if trials == 0 and stats.rate_limited > 0:
            logger.warning("Batch %d produced nothing under throttling — stopping", batch_index)
            break

    progress = silver_stats(db, signature)
    return ExtractionRunResult(
        signature=signature,
        trials_written=trials_written,
        criteria_written=criteria_written,
        succeeded=stats.succeeded,
        failed=stats.failed,
        rate_limited=stats.rate_limited,
        remaining=progress["total"] - progress["trials"],
    )
