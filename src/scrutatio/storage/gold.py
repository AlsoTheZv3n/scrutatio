"""Gold layer: embeddings over criteria, and the search that reads them.

Vector search is not a service here. DuckDB has fixed-size ``FLOAT[N]`` columns
and a native ``array_cosine_similarity``, so the index is a table and top-K is a
query. At ~300,000 vectors that is the right shape — a managed index would add a
component, a sync mechanism and a failure mode for no measurable gain.

**Criteria are embedded individually, not whole trials.** That is the point of
the whole pipeline: a patient phrase like "progression on osimertinib" has to
reach a criterion that says "progression on or after a third-generation EGFR
TKI", and it can only do that if the criterion is its own vector. A trial's
score is then the score of its best-matching criterion, which also yields the
reason it matched — useful directly in the UI.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Final

import numpy as np
import pyarrow as pa

from scrutatio.config import EMBEDDING_DIMENSIONS
from scrutatio.storage.silver import SILVER_TABLE

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

GOLD_TABLE: Final = "gold_embeddings"


def ensure_gold(db: duckdb.DuckDBPyConnection) -> None:
    """Create the Gold table if it does not exist."""
    db.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {GOLD_TABLE} (
            nct_id VARCHAR NOT NULL,
            signature VARCHAR NOT NULL,
            ordinal INTEGER NOT NULL,
            embedding_model VARCHAR NOT NULL,
            embedding FLOAT[{EMBEDDING_DIMENSIONS}] NOT NULL,
            embedded_at TIMESTAMP NOT NULL,
            PRIMARY KEY (nct_id, signature, ordinal, embedding_model)
        )
        """
    )


def pending_criteria(
    db: duckdb.DuckDBPyConnection,
    signature: str,
    model: str,
    *,
    limit: int | None = None,
) -> list[tuple[str, int, str]]:
    """Criteria with no vector yet, as ``(nct_id, ordinal, text)``.

    Keyed on the extraction signature *and* the embedding model, so changing
    either re-queues exactly the affected work — the same mechanism Silver uses,
    for the same reason.
    """
    params: list[object] = [signature, model, signature]
    limit_clause = ""
    if limit is not None:
        limit_clause = "LIMIT ?"
        params.append(int(limit))

    rows = db.execute(
        f"""
        SELECT s.nct_id, s.ordinal, s.text
        FROM {SILVER_TABLE} s
        WHERE NOT EXISTS (
            SELECT 1 FROM {GOLD_TABLE} g
            WHERE g.nct_id = s.nct_id AND g.ordinal = s.ordinal
              AND g.signature = ? AND g.embedding_model = ?
        )
          AND s.signature = ?
        ORDER BY s.nct_id, s.ordinal
        {limit_clause}
        """,  # noqa: S608 - table names are module constants
        params,
    ).fetchall()
    return [(str(r[0]), int(r[1]), str(r[2])) for r in rows]


def write_embeddings(
    db: duckdb.DuckDBPyConnection,
    rows: Iterable[tuple[str, int, Sequence[float]]],
    *,
    signature: str,
    model: str,
) -> int:
    """Land ``(nct_id, ordinal, vector)`` triples. Returns the row count.

    Bulk-loads through an Arrow table rather than ``executemany``. That is not a
    micro-optimisation: profiled on a real batch of 2,048 criteria, the write was
    **95% of the wall clock** — 91.8 seconds against 4.2 seconds of GPU encoding.
    Binding 2,048 rows of 1,024 floats one at a time, each with a four-column
    primary-key lookup, dominated everything the accelerator did.

    Via Arrow the same batch writes in about a second and a half, which puts the
    GPU back where it belongs as the limiting factor. ``ON CONFLICT`` survives the
    change and still costs almost nothing, so re-running a batch remains
    idempotent.
    """
    materialised = list(rows)
    if not materialised:
        return 0

    vectors = np.asarray([r[2] for r in materialised], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIMENSIONS:
        got = f"{vectors.shape[1]} dimensions" if vectors.ndim == 2 else f"shape {vectors.shape}"
        msg = (
            f"Vectors have {got} per row, but the column is FLOAT[{EMBEDDING_DIMENSIONS}] "
            f"(first row: {materialised[0][0]}#{materialised[0][1]})"
        )
        raise ValueError(msg)

    staged = pa.table(
        {
            "nct_id": pa.array([r[0] for r in materialised]),
            "signature": pa.array([signature] * len(materialised)),
            "ordinal": pa.array([r[1] for r in materialised], type=pa.int32()),
            "embedding_model": pa.array([model] * len(materialised)),
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.reshape(-1)), EMBEDDING_DIMENSIONS
            ),
        }
    )
    # Registered by name rather than left to DuckDB's replacement scan, which
    # would resolve `staged` out of the local scope — that works but reads as an
    # unused variable and depends on frame introspection.
    db.register("_staged_vectors", staged)
    try:
        db.execute(
            f"""
            INSERT INTO {GOLD_TABLE}
                (nct_id, signature, ordinal, embedding_model, embedding, embedded_at)
            SELECT nct_id, signature, ordinal, embedding_model, embedding, current_timestamp
            FROM _staged_vectors
            ON CONFLICT (nct_id, signature, ordinal, embedding_model) DO UPDATE SET
                embedding = excluded.embedding,
                embedded_at = excluded.embedded_at
            """  # noqa: S608
        )
    finally:
        db.unregister("_staged_vectors")
    return len(materialised)


def search_criteria(
    db: duckdb.DuckDBPyConnection,
    query: Sequence[float],
    *,
    signature: str,
    model: str,
    k: int = 10,
) -> list[tuple[str, float, int, str]]:
    """Top-K trials for a query vector.

    Returns ``(nct_id, score, ordinal, text)`` — the trial, how well it matched,
    and *which criterion* did the matching. The last two are what makes the
    result explainable rather than a bare ranking.

    A trial scores as its single best-matching criterion. Averaging would punish
    trials with many administrative criteria (consent, compliance) that no
    patient description will ever resemble.
    """
    if len(query) != EMBEDDING_DIMENSIONS:
        msg = f"Query vector has {len(query)} dimensions, expected {EMBEDDING_DIMENSIONS}"
        raise ValueError(msg)

    rows = db.execute(
        f"""
        WITH scored AS (
            SELECT g.nct_id, g.ordinal, s.text,
                   array_cosine_similarity(g.embedding, ?::FLOAT[{EMBEDDING_DIMENSIONS}]) AS score
            FROM {GOLD_TABLE} g
            JOIN {SILVER_TABLE} s
              ON s.nct_id = g.nct_id AND s.ordinal = g.ordinal AND s.signature = g.signature
            WHERE g.signature = ? AND g.embedding_model = ?
        ),
        best AS (
            SELECT *, row_number() OVER (PARTITION BY nct_id ORDER BY score DESC) AS rn
            FROM scored
        )
        SELECT nct_id, score, ordinal, text
        FROM best WHERE rn = 1
        ORDER BY score DESC
        LIMIT ?
        """,  # noqa: S608
        [list(query), signature, model, int(k)],
    ).fetchall()
    return [(str(r[0]), float(r[1]), int(r[2]), str(r[3])) for r in rows]


def gold_stats(db: duckdb.DuckDBPyConnection, signature: str, model: str) -> dict[str, int]:
    """Progress counters for the current signature and model."""
    row = db.execute(
        f"""
        SELECT
            (SELECT count(*) FROM {GOLD_TABLE} WHERE signature = ? AND embedding_model = ?),
            (SELECT count(DISTINCT nct_id) FROM {GOLD_TABLE}
             WHERE signature = ? AND embedding_model = ?),
            (SELECT count(*) FROM {SILVER_TABLE} WHERE signature = ?)
        """,  # noqa: S608
        [signature, model, signature, model, signature],
    ).fetchone()
    if not row:
        return {"vectors": 0, "trials": 0, "criteria": 0}
    vectors, trials, criteria = (int(v or 0) for v in row)
    return {"vectors": vectors, "trials": trials, "criteria": criteria}
