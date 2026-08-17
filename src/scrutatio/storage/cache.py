"""Cached per-trial verdicts.

Judging one trial costs a model call of roughly ten seconds. At K=10 that is the
entire latency budget, and the same query asked twice would pay it twice. So the
verdicts are stored — and keyed on everything that could change them.

Four parts to the key, each for a reason:

* ``query_hash`` — the patient description. Different patient, different answer.
* ``nct_id`` — verdicts are per trial, so a later query that retrieves an
  overlapping set of trials reuses the ones it has seen.
* ``signature`` — the extraction signature. If the criteria change, verdicts
  about the old criteria are about text that no longer exists.
* ``prompt_version`` — the judge model, prompt and schema. Editing the prompt
  must not leave answers produced by the previous one sitting in the cache
  looking current.

That last pair is the same mechanism Silver and Gold use. It has caught a silent
mismatch three times now, so it is applied here before it can.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

from scrutatio.matching.schema import CriterionVerdict

if TYPE_CHECKING:
    from collections.abc import Sequence

    import duckdb

logger = logging.getLogger(__name__)

CACHE_TABLE: Final = "match_cache"

_VERDICTS = TypeAdapter(list[CriterionVerdict])


def query_hash(text: str) -> str:
    """Stable key for a patient description.

    Whitespace-normalised and lowercased: the same vignette pasted with a
    different line wrap is the same question, and paying twice for it would be a
    surprise rather than a feature.
    """
    normalised = " ".join(text.split()).lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:32]


def ensure_cache(db: duckdb.DuckDBPyConnection) -> None:
    """Create the verdict cache if it does not exist."""
    db.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {CACHE_TABLE} (
            query_hash VARCHAR NOT NULL,
            nct_id VARCHAR NOT NULL,
            signature VARCHAR NOT NULL,
            prompt_version VARCHAR NOT NULL,
            verdicts VARCHAR NOT NULL,
            cached_at TIMESTAMP NOT NULL,
            PRIMARY KEY (query_hash, nct_id, signature, prompt_version)
        )
        """
    )


def cached_verdicts(
    db: duckdb.DuckDBPyConnection,
    query_key: str,
    nct_ids: Sequence[str],
    *,
    signature: str,
    prompt_version: str,
) -> dict[str, list[CriterionVerdict]]:
    """Verdicts already computed for this query, per trial."""
    if not nct_ids:
        return {}

    rows = db.execute(
        f"""
        SELECT nct_id, verdicts FROM {CACHE_TABLE}
        WHERE query_hash = ? AND signature = ? AND prompt_version = ?
          AND nct_id IN (SELECT unnest(?::VARCHAR[]))
        """,  # noqa: S608 - table name is a module constant
        [query_key, signature, prompt_version, list(nct_ids)],
    ).fetchall()

    out: dict[str, list[CriterionVerdict]] = {}
    for identifier, payload in rows:
        try:
            out[str(identifier)] = _VERDICTS.validate_json(payload)
        except ValueError:  # pragma: no cover - a schema change invalidates via the key
            logger.warning("Discarding unreadable cache entry for %s", identifier)
    return out


def store_verdicts(
    db: duckdb.DuckDBPyConnection,
    query_key: str,
    nct_id: str,
    verdicts: Sequence[CriterionVerdict],
    *,
    signature: str,
    prompt_version: str,
) -> None:
    """Record one trial's verdicts for this query."""
    db.execute(
        f"""
        INSERT INTO {CACHE_TABLE}
            (query_hash, nct_id, signature, prompt_version, verdicts, cached_at)
        VALUES (?, ?, ?, ?, ?, current_timestamp)
        ON CONFLICT (query_hash, nct_id, signature, prompt_version) DO UPDATE SET
            verdicts = excluded.verdicts,
            cached_at = excluded.cached_at
        """,  # noqa: S608
        [
            query_key,
            nct_id,
            signature,
            prompt_version,
            json.dumps([v.model_dump() for v in verdicts], ensure_ascii=False),
        ],
    )


def cache_stats(db: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """How much is cached, for the status view."""
    row = db.execute(
        f"SELECT count(*), count(DISTINCT query_hash) FROM {CACHE_TABLE}"  # noqa: S608
    ).fetchone()
    if not row:
        return {"entries": 0, "queries": 0}
    return {"entries": int(row[0] or 0), "queries": int(row[1] or 0)}
