"""Concurrent extraction over many trials.

Concurrency is a precondition, not an optimisation: a criteria-heavy trial takes
tens of seconds, and 11,200 of them in sequence is measured in days. OpenRouter
publishes no concurrency limit for paid accounts, so the runner reports what it
actually hit — successes, failures, and how often the endpoint pushed back with
429 — because that telemetry is the only available answer to "how fast may we
go".

Sustained measurement, 16 workers: 895 trials/hour, zero 429s over 505 calls.
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
# OpenRouter publishes no concurrency limit for paid accounts and reroutes around
# provider throttling, so the safe number is unknown until the calibration gate
# measures it under sustained load. Start low and raise on evidence: the previous
# platform was tuned *downward* from 8 and lost a night to it.
DEFAULT_WORKERS = 4


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
        except Exception as exc:
            # One trial must never end the run. This is not defensive padding:
            # a JSONDecodeError from a malformed response body propagated out of
            # a worker, through future.result(), and terminated a pass that was
            # three batches deep. Anything unforeseen belongs in the failure
            # table alongside the foreseen kinds, not in a traceback.
            tracker.record(ok=False, throttled=False)
            logger.exception("Unexpected error extracting %s", nct_id)
            detail = f"{type(exc).__name__}: {exc}"
            return ExtractionOutcome(nct_id=nct_id, result=None, error=detail[:500])
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
