"""Silver layer: eligibility prose turned into typed predicates.

Resumability is the design centre, not an add-on. A full pass over 11,200
trials cannot complete in one sitting under the Free Edition quota, so the run
has to survive being stopped and restarted repeatedly without redoing work or
silently skipping trials.

That requires knowing *what* an extraction was produced by. The taxonomy change
on 2026-08-15 — nine criterion kinds to twenty — invalidated every extraction
made before it, and nothing in the data said so. Every row therefore carries an
**extraction signature**: a hash of the model endpoint, the system prompt and
the JSON schema. Change any of them and the affected trials become pending
again automatically, instead of leaving a silently mixed table.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Final

from scrutatio.config import Settings, get_settings
from scrutatio.extraction.client import SYSTEM_PROMPT
from scrutatio.extraction.schema import ExtractedEligibility, flatten_schema
from scrutatio.storage.bronze import (
    BRONZE_TABLE,
    CATALOG,
    SCHEMA,
    VOLUME,
    _validate_batch,
)
from scrutatio.storage.sql import SqlClient

if TYPE_CHECKING:
    from scrutatio.extraction.runner import ExtractionOutcome

logger = logging.getLogger(__name__)

SILVER_TABLE: Final = f"{CATALOG}.{SCHEMA}.silver_criteria"
FAILURES_TABLE: Final = f"{CATALOG}.{SCHEMA}.silver_failures"
_STAGING_TABLE: Final = f"{CATALOG}.{SCHEMA}._silver_staging"

ELIGIBILITY_PATH: Final = "$.protocolSection.eligibilityModule.eligibilityCriteria"


def extraction_signature(settings: Settings | None = None) -> str:
    """Short hash identifying what produced an extraction.

    Covers the model endpoint, the system prompt and the JSON schema — the three
    inputs that change what comes out. Rows carrying a different signature are
    treated as absent, so a prompt or taxonomy change re-queues exactly the
    affected work with no manual bookkeeping.
    """
    cfg = settings or get_settings()
    material = json.dumps(
        {
            "endpoint": cfg.chat_endpoint,
            "system_prompt": SYSTEM_PROMPT,
            "schema": flatten_schema(ExtractedEligibility),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def ensure_silver(sql: SqlClient) -> None:
    """Create the Silver and failure tables if they do not exist."""
    sql.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SILVER_TABLE} (
            nct_id STRING NOT NULL,
            signature STRING NOT NULL,
            ordinal INT NOT NULL,
            criterion_id STRING,
            text STRING NOT NULL,
            kind STRING NOT NULL,
            is_exclusion BOOLEAN NOT NULL,
            extracted_at TIMESTAMP NOT NULL
        )
        USING DELTA
        TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """
    )
    # Failures are recorded rather than retried forever: a trial that fails on
    # every pass would otherwise stall the run at the same place each restart.
    sql.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {FAILURES_TABLE} (
            nct_id STRING NOT NULL,
            signature STRING NOT NULL,
            error STRING,
            attempts INT NOT NULL,
            last_failed_at TIMESTAMP NOT NULL
        )
        USING DELTA
        """
    )


def pending_trials(
    sql: SqlClient,
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
    # S608: `signature` is a hex digest from extraction_signature(); the numeric
    # bounds are int-cast. No caller-supplied text reaches the statement.
    failure_clause = (
        ""
        if retry_failed
        else f"""
        AND NOT EXISTS (
            SELECT 1 FROM {FAILURES_TABLE} f
            WHERE f.nct_id = b.nct_id
              AND f.signature = '{signature}'
              AND f.attempts >= {int(max_attempts)}
        )
        """  # noqa: S608
    )
    rows = sql.execute(
        f"""
        SELECT b.nct_id, get_json_object(b.raw, '{ELIGIBILITY_PATH}') AS eligibility
        FROM {BRONZE_TABLE} b
        WHERE get_json_object(b.raw, '{ELIGIBILITY_PATH}') IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM {SILVER_TABLE} s
              WHERE s.nct_id = b.nct_id AND s.signature = '{signature}'
          )
          {failure_clause}
        ORDER BY b.nct_id
        {f"LIMIT {int(limit)}" if limit else ""}
        """  # noqa: S608 - signature is a hex digest, limits are int-cast
    )
    return [(str(r[0]), str(r[1])) for r in rows if r[1]]


def _to_ndjson(outcomes: Iterable[ExtractionOutcome], signature: str) -> tuple[bytes, int, int]:
    """Serialise successful outcomes. Returns (payload, criteria rows, trials)."""
    lines: list[str] = []
    trials = 0
    for outcome in outcomes:
        if not outcome.ok or outcome.result is None:
            continue
        trials += 1
        for ordinal, criterion in enumerate(outcome.result.criteria):
            lines.append(
                json.dumps(
                    {
                        "nct_id": outcome.nct_id,
                        "signature": signature,
                        # Numbers and booleans are written as JSON *strings* so
                        # COPY INTO infers every staging column as STRING. A raw
                        # `0` is inferred as bigint and fails the STRING staging
                        # column with DELTA_FAILED_TO_MERGE_FIELDS. Casting
                        # happens on the way into the typed target table.
                        "ordinal": str(ordinal),
                        "criterion_id": criterion.criterion_id,
                        "text": criterion.text,
                        "kind": criterion.kind,
                        "is_exclusion": str(criterion.is_exclusion).lower(),
                    },
                    ensure_ascii=False,
                )
            )
    payload = gzip.compress(("\n".join(lines) + "\n").encode("utf-8"))
    return payload, len(lines), trials


def write_silver(
    sql: SqlClient,
    outcomes: Iterable[ExtractionOutcome],
    *,
    batch: str,
    signature: str,
) -> tuple[int, int]:
    """Land one batch of extractions. Returns ``(criteria_rows, trials)``.

    Re-extracting a trial replaces its rows rather than merging them: criterion
    ids are model-generated and not stable across runs, so a MERGE would leave
    orphaned rows from the previous shape.
    """
    _validate_batch(batch)
    materialised = list(outcomes)

    _record_failures(sql, materialised, signature)

    payload, rows, trials = _to_ndjson(materialised, signature)
    if trials == 0:
        return 0, 0

    volume_path = f"{CATALOG}/{SCHEMA}/{VOLUME}/{batch}.ndjson.gz"
    sql.upload_file(volume_path, payload)

    sql.execute(f"DROP TABLE IF EXISTS {_STAGING_TABLE}")
    sql.execute(
        f"""
        CREATE TABLE {_STAGING_TABLE} (
            nct_id STRING, signature STRING, ordinal STRING, criterion_id STRING,
            text STRING, kind STRING, is_exclusion STRING
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
    # Replace-per-trial, scoped to this batch and signature.
    sql.execute(
        f"""
        DELETE FROM {SILVER_TABLE}
        WHERE signature = '{signature}'
          AND nct_id IN (SELECT DISTINCT nct_id FROM {_STAGING_TABLE})
        """  # noqa: S608
    )
    sql.execute(
        f"""
        INSERT INTO {SILVER_TABLE}
        SELECT nct_id, signature, CAST(ordinal AS INT), criterion_id, text, kind,
               CAST(is_exclusion AS BOOLEAN), current_timestamp()
        FROM {_STAGING_TABLE}
        """  # noqa: S608
    )
    sql.execute(f"DROP TABLE IF EXISTS {_STAGING_TABLE}")

    # A trial that now succeeded must stop counting as failed.
    sql.execute(
        f"""
        DELETE FROM {FAILURES_TABLE}
        WHERE signature = '{signature}'
          AND nct_id IN ({_quote_list(o.nct_id for o in materialised if o.ok)})
        """  # noqa: S608
    )

    logger.info("Landed %d criteria from %d trials (batch %s)", rows, trials, batch)
    return rows, trials


def _quote_list(values: Iterable[str]) -> str:
    """Quote NCT ids for an IN clause. Ids are validated by the generated model."""
    quoted = [f"'{v}'" for v in values if v.replace("NCT", "").isdigit()]
    return ", ".join(quoted) if quoted else "''"


def _is_throttled(outcome: ExtractionOutcome) -> bool:
    """True when the trial failed because the endpoint was rate limiting."""
    return "429" in (outcome.error or "")


def _record_failures(sql: SqlClient, outcomes: list[ExtractionOutcome], signature: str) -> None:
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

    values = ", ".join(
        f"('{o.nct_id}', '{signature}', {_sql_string(o.error or 'unknown')})" for o in failed
    )
    sql.execute(
        f"""
        MERGE INTO {FAILURES_TABLE} AS target
        USING (SELECT * FROM VALUES {values} AS t(nct_id, signature, error)) AS source
        ON target.nct_id = source.nct_id AND target.signature = source.signature
        WHEN MATCHED THEN UPDATE SET
            target.attempts = target.attempts + 1,
            target.error = source.error,
            target.last_failed_at = current_timestamp()
        WHEN NOT MATCHED THEN INSERT (nct_id, signature, error, attempts, last_failed_at)
        VALUES (source.nct_id, source.signature, source.error, 1, current_timestamp())
        """  # noqa: S608
    )
    logger.warning("Recorded %d extraction failures", len(failed))


def _sql_string(value: str) -> str:
    """Quote a string literal, escaping embedded single quotes."""
    escaped = value.replace("'", "''")[:400]
    return f"'{escaped}'"


def silver_stats(sql: SqlClient, signature: str) -> dict[str, int]:
    """Progress counters for the current signature."""
    rows = sql.execute(
        f"""
        SELECT
            (SELECT count(DISTINCT nct_id) FROM {SILVER_TABLE} WHERE signature = '{signature}'),
            (SELECT count(*) FROM {SILVER_TABLE} WHERE signature = '{signature}'),
            (SELECT count(*) FROM {FAILURES_TABLE} WHERE signature = '{signature}'),
            (SELECT count(*) FROM {BRONZE_TABLE}
             WHERE get_json_object(raw, '{ELIGIBILITY_PATH}') IS NOT NULL)
        """  # noqa: S608
    )
    if not rows:
        return {"trials": 0, "criteria": 0, "failed": 0, "total": 0}
    trials, criteria, failed, total = (int(v or 0) for v in rows[0])
    return {"trials": trials, "criteria": criteria, "failed": failed, "total": total}
