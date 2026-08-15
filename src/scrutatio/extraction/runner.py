"""Concurrent extraction over many trials.

Serial extraction was measured at ~27s for a criteria-heavy trial, which puts a
full 11,200-study pass at roughly 84 hours. Concurrency is therefore not an
optimisation but a precondition — and since the Free Edition rate limits are
undocumented, the runner has to discover them at runtime rather than assume a
number.

It therefore reports what it hit: successes, failures, and how often the
endpoint pushed back with 429. That telemetry is the answer to "what is the
quota", which no Databricks page states.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from scrutatio.extraction.client import EligibilityExtractor, ExtractionError

if TYPE_CHECKING:
    from scrutatio.extraction.schema import ExtractedEligibility

logger = logging.getLogger(__name__)

# httpx.Client is safe to share across threads; the connection pool serialises
# access. A separate client per worker would multiply TLS handshakes for nothing.
#
# Three, not eight. Measured against Azure Databricks Premium (2026-08-15,
# 8 requests per setting): 1-2 workers produced zero 429s, 4 workers failed
# 2 of 8, and 8 workers failed 6 of 8. The endpoint reserves `max_tokens`
# before admitting a request, so concurrency multiplies the reservation and
# the limit is hit long before the model is busy. Published Enterprise-tier
# limits do not apply to Premium.
DEFAULT_WORKERS = 3


@dataclass
class RunStats:
    """Observed behaviour of one extraction run."""

    succeeded: int = 0
    failed: int = 0
    rate_limited: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, *, ok: bool, throttled: bool) -> None:
        with self._lock:
            if ok:
                self.succeeded += 1
            else:
                self.failed += 1
            if throttled:
                self.rate_limited += 1

    @property
    def attempted(self) -> int:
        return self.succeeded + self.failed


@dataclass(frozen=True)
class ExtractionOutcome:
    """Result for one trial. ``error`` is set exactly when ``result`` is None."""

    nct_id: str
    result: ExtractedEligibility | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.result is not None


def extract_many(
    extractor: EligibilityExtractor,
    items: Iterable[tuple[str, str]],
    *,
    max_workers: int = DEFAULT_WORKERS,
    stats: RunStats | None = None,
    on_done: Callable[[ExtractionOutcome], None] | None = None,
) -> Iterator[ExtractionOutcome]:
    """Extract ``(nct_id, eligibility_text)`` pairs concurrently.

    Yields outcomes as they complete, in completion order rather than input
    order. A failure on one trial never aborts the run: it is reported as an
    outcome with ``error`` set, so a partial pass still lands everything that
    worked and the failures can be retried by NCT id.

    The input is consumed lazily and only ``max_workers * 2`` futures are kept in
    flight, so this streams over 11,200 trials without materialising them.
    """
    tracker = stats if stats is not None else RunStats()
    source = iter(items)
    pending: set[Future[ExtractionOutcome]] = set()

    def work(nct_id: str, text: str) -> ExtractionOutcome:
        try:
            result = extractor.extract(text)
        except ExtractionError as exc:
            throttled = "429" in str(exc)
            tracker.record(ok=False, throttled=throttled)
            logger.warning("Extraction failed for %s: %s", nct_id, str(exc)[:160])
            return ExtractionOutcome(nct_id=nct_id, result=None, error=str(exc)[:500])
        tracker.record(ok=True, throttled=False)
        return ExtractionOutcome(nct_id=nct_id, result=result)

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="extract") as pool:

        def submit_next() -> bool:
            item = next(source, None)
            if item is None:
                return False
            pending.add(pool.submit(work, item[0], item[1]))
            return True

        for _ in range(max_workers * 2):
            if not submit_next():
                break

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                outcome = future.result()
                if on_done is not None:
                    on_done(outcome)
                yield outcome
                submit_next()
