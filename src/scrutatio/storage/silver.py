"""Silver layer: eligibility prose turned into typed predicates.

Resumability is the design centre, not an add-on. A full pass over 11,200 trials
runs against an external API, so it has to survive being stopped and restarted
without redoing work or silently skipping trials.

That requires knowing *what* an extraction was produced by. The taxonomy change
on 2026-08-15 — nine criterion kinds to twenty — invalidated every extraction
made before it, and nothing in the data said so. Every row therefore carries an
**extraction signature**: a hash of the model, the system prompt and the JSON
schema. Change any of them and the affected trials become pending again
automatically, instead of leaving a silently mixed table. It is also what keeps
rows produced on different platforms from ever mixing.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Final

from scrutatio.config import Settings, get_settings
from scrutatio.extraction.client import SYSTEM_PROMPT
from scrutatio.extraction.schema import ExtractedEligibility, flatten_schema
from scrutatio.storage.bronze import BRONZE_TABLE
from scrutatio.utils.hashing import stable_digest

if TYPE_CHECKING:
    import duckdb

    from scrutatio.extraction.runner import ExtractionOutcome

logger = logging.getLogger(__name__)

SILVER_TABLE: Final = "silver_criteria"
FAILURES_TABLE: Final = "silver_failures"

ELIGIBILITY_PATH: Final = "$.protocolSection.eligibilityModule.eligibilityCriteria"


def extraction_signature(settings: Settings | None = None) -> str:
    """Short hash identifying what produced an extraction.

    Covers the model, the system prompt and the JSON schema — the three inputs
    that change what comes out. Rows carrying a different signature are treated
    as absent, so a prompt, model or taxonomy change re-queues exactly the
    affected work with no manual bookkeeping.
    """
    cfg = settings or get_settings()
    return stable_digest(
        {
            "model": cfg.extraction_model,
            # Not cosmetic: with reasoning on the same model produced 3,355 output
            # tokens per trial against 1,743 with it off. Anything that changes
            # what comes out belongs in the hash.
            "reasoning": cfg.extraction_reasoning,
            "system_prompt": SYSTEM_PROMPT,
            "schema": flatten_schema(ExtractedEligibility),
        }
    )


def ensure_silver(db: duckdb.DuckDBPyConnection) -> None:
    """Create the Silver and failure tables if they do not exist."""
    db.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SILVER_TABLE} (
            nct_id VARCHAR NOT NULL,
            signature VARCHAR NOT NULL,
            ordinal INTEGER NOT NULL,
            criterion_id VARCHAR,
            text VARCHAR NOT NULL,
            kind VARCHAR NOT NULL,
            is_exclusion BOOLEAN NOT NULL,
            extracted_at TIMESTAMP NOT NULL,
            PRIMARY KEY (nct_id, signature, ordinal)
        )
        """
    )
    # Failures are recorded rather than retried forever: a trial that fails on
    # every pass would otherwise stall the run at the same place each restart.
    db.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {FAILURES_TABLE} (
            nct_id VARCHAR NOT NULL,
            signature VARCHAR NOT NULL,
            error VARCHAR,
            attempts INTEGER NOT NULL,
            last_failed_at TIMESTAMP NOT NULL,
            PRIMARY KEY (nct_id, signature)
        )
        """
    )


def pending_trials(
    db: duckdb.DuckDBPyConnection,
    signature: str,
    *,
    limit: int | None = None,
    retry_failed: bool = False,
    max_attempts: int = 3,
) -> list[tuple[str, str]]:
    """Trials still needing extraction, as ``(nct_id, eligibility_text)``.

    A trial is pending when Bronze has it but Silver holds no row for it under
    the current signature. Trials that failed repeatedly are excluded unless
    ``retry_failed`` is set, so a restart makes forward progress instead of
    re-attempting the same broken inputs.
    """
    params: list[object] = [ELIGIBILITY_PATH, ELIGIBILITY_PATH, signature]
    failure_clause = ""
    if not retry_failed:
        failure_clause = f"""
        AND NOT EXISTS (
            SELECT 1 FROM {FAILURES_TABLE} f
            WHERE f.nct_id = b.nct_id AND f.signature = ? AND f.attempts >= ?
        )
        """  # noqa: S608 - table names are module constants
        params += [signature, int(max_attempts)]

    limit_clause = ""
    if limit is not None:
        limit_clause = "LIMIT ?"
        params.append(int(limit))

    rows = db.execute(
        f"""
        SELECT b.nct_id, json_extract_string(b.raw, ?) AS eligibility
        FROM {BRONZE_TABLE} b
        WHERE json_extract_string(b.raw, ?) IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM {SILVER_TABLE} s
              WHERE s.nct_id = b.nct_id AND s.signature = ?
          )
          {failure_clause}
        ORDER BY b.nct_id
        {limit_clause}
        """,  # noqa: S608
        params,
    ).fetchall()
    return [(str(r[0]), str(r[1])) for r in rows if r[1]]


def write_silver(
    db: duckdb.DuckDBPyConnection,
    outcomes: Iterable[ExtractionOutcome],
    *,
    batch: str = "-",
    signature: str,
) -> tuple[int, int]:
    """Land one batch of extractions. Returns ``(criteria_rows, trials)``.

    Re-extracting a trial replaces its rows rather than merging them: criterion
    ids are model-generated and not stable across runs, so a merge would leave
    orphaned rows from the previous shape.
    """
    materialised = list(outcomes)
    _record_failures(db, materialised, signature)

    succeeded = [o for o in materialised if o.ok and o.result is not None]
    if not succeeded:
        return 0, 0

    rows = [
        (o.nct_id, signature, ordinal, c.criterion_id, c.text, c.kind, c.is_exclusion)
        for o in succeeded
        for ordinal, c in enumerate(o.result.criteria)  # type: ignore[union-attr]
    ]

    # Replace per trial, scoped to this signature.
    db.executemany(
        f"DELETE FROM {SILVER_TABLE} WHERE signature = ? AND nct_id = ?",  # noqa: S608
        [(signature, o.nct_id) for o in succeeded],
    )
    if rows:
        db.executemany(
            f"""
            INSERT INTO {SILVER_TABLE}
            (nct_id, signature, ordinal, criterion_id, text, kind, is_exclusion, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, current_timestamp)
            """,  # noqa: S608
            rows,
        )

    # A trial that now succeeded must stop counting as failed.
    db.executemany(
        f"DELETE FROM {FAILURES_TABLE} WHERE signature = ? AND nct_id = ?",  # noqa: S608
        [(signature, o.nct_id) for o in succeeded],
    )

    logger.info("Landed %d criteria from %d trials (batch %s)", len(rows), len(succeeded), batch)
    return len(rows), len(succeeded)


def _is_throttled(outcome: ExtractionOutcome) -> bool:
    """True when the trial failed because the endpoint was rate limiting."""
    return "429" in (outcome.error or "")


def _record_failures(
    db: duckdb.DuckDBPyConnection, outcomes: list[ExtractionOutcome], signature: str
) -> None:
    """Increment the attempt counter for trials that genuinely failed.

    Throttled trials are deliberately **not** recorded. A 429 says something
    about the moment, not about the trial: the input is fine and will extract
    once capacity returns. Counting throttling against the attempt budget
    retires good trials from the corpus permanently — measured at three
    exclusions in the first minute of a run, which over a night would have
    silently removed hundreds of trials that were never examined.
    """
    throttled = sum(1 for o in outcomes if not o.ok and _is_throttled(o))
    if throttled:
        logger.info("%d trials throttled — left pending, not counted against attempts", throttled)

    failed = [o for o in outcomes if not o.ok and not _is_throttled(o)]
    if not failed:
        return

    db.executemany(
        f"""
        INSERT INTO {FAILURES_TABLE} (nct_id, signature, error, attempts, last_failed_at)
        VALUES (?, ?, ?, 1, current_timestamp)
        ON CONFLICT (nct_id, signature) DO UPDATE SET
            attempts = {FAILURES_TABLE}.attempts + 1,
            error = excluded.error,
            last_failed_at = excluded.last_failed_at
        """,  # noqa: S608
        [(o.nct_id, signature, (o.error or "unknown")[:400]) for o in failed],
    )
    logger.warning("Recorded %d extraction failures", len(failed))


def trial_criteria(
    db: duckdb.DuckDBPyConnection, nct_ids: Sequence[str], signature: str
) -> dict[str, list[tuple[int, str, str, bool]]]:
    """Every criterion of the given trials, as ``(ordinal, text, kind, is_exclusion)``.

    Retrieval finds a trial through its single best-matching criterion, but a
    verdict has to cover all of them — a patient can be excluded by a criterion
    that looks nothing like their description, which is the whole reason the
    ranking is not the answer.
    """
    if not nct_ids:
        return {}

    rows = db.execute(
        f"""
        SELECT nct_id, ordinal, text, kind, is_exclusion
        FROM {SILVER_TABLE}
        WHERE signature = ? AND nct_id IN (SELECT unnest(?::VARCHAR[]))
        ORDER BY nct_id, ordinal
        """,  # noqa: S608 - table name is a module constant
        [signature, list(nct_ids)],
    ).fetchall()

    out: dict[str, list[tuple[int, str, str, bool]]] = {}
    for identifier, ordinal, text, kind, is_exclusion in rows:
        out.setdefault(str(identifier), []).append(
            (int(ordinal), str(text), str(kind), bool(is_exclusion))
        )
    return out


def silver_stats(db: duckdb.DuckDBPyConnection, signature: str) -> dict[str, int]:
    """Progress counters for the current signature."""
    row = db.execute(
        f"""
        SELECT
            (SELECT count(DISTINCT nct_id) FROM {SILVER_TABLE} WHERE signature = ?),
            (SELECT count(*) FROM {SILVER_TABLE} WHERE signature = ?),
            (SELECT count(*) FROM {FAILURES_TABLE} WHERE signature = ?),
            (SELECT count(*) FROM {BRONZE_TABLE}
             WHERE json_extract_string(raw, ?) IS NOT NULL)
        """,  # noqa: S608
        [signature, signature, signature, ELIGIBILITY_PATH],
    ).fetchone()
    if not row:
        return {"trials": 0, "criteria": 0, "failed": 0, "total": 0}
    trials, criteria, failed, total = (int(v or 0) for v in row)
    return {"trials": trials, "criteria": criteria, "failed": failed, "total": total}
