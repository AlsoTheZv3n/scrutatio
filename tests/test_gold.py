"""Gold tests: embeddings and the search that reads them.

Run against a real in-memory DuckDB with constructed unit vectors, so cosine
similarities are exact and the assertions are about ranking rather than about a
model's behaviour. Nothing here needs torch — the storage layer is testable
without the embedding stack installed, which is why they are separate modules.
"""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import pytest

from scrutatio.config import EMBEDDING_DIMENSIONS
from scrutatio.storage.db import IN_MEMORY, connect
from scrutatio.storage.gold import (
    GOLD_TABLE,
    ensure_gold,
    gold_stats,
    pending_criteria,
    search_criteria,
    write_embeddings,
)
from scrutatio.storage.silver import SILVER_TABLE, ensure_silver

SIG = "abc123def4567890"
OTHER_SIG = "0000111122223333"
MODEL = "BAAI/bge-large-en-v1.5"
OTHER_MODEL = "intfloat/e5-large-v2"


def _unit(axis: int) -> list[float]:
    """A unit vector along one axis. Cosine similarity between two of these is
    1.0 when the axis matches and 0.0 when it does not — so ranking assertions
    are exact rather than approximate."""
    vec = [0.0] * EMBEDDING_DIMENSIONS
    vec[axis] = 1.0
    return vec


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = connect(IN_MEMORY)
    ensure_silver(conn)
    ensure_gold(conn)
    try:
        yield conn
    finally:
        conn.close()


def _land_criterion(
    db: duckdb.DuckDBPyConnection,
    nct: str,
    ordinal: int,
    text: str,
    signature: str = SIG,
) -> None:
    db.execute(
        f"INSERT INTO {SILVER_TABLE} VALUES (?, ?, ?, 'c', ?, 'condition', false, current_timestamp)",  # noqa: S608, E501
        [nct, signature, ordinal, text],
    )


class TestSchemaSetup:
    def test_ddl_is_idempotent(self, db: duckdb.DuckDBPyConnection) -> None:
        ensure_gold(db)
        ensure_gold(db)
        assert gold_stats(db, SIG, MODEL)["vectors"] == 0


class TestPendingCriteria:
    def test_returns_criteria_without_a_vector(self, db: duckdb.DuckDBPyConnection) -> None:
        _land_criterion(db, "NCT00000001", 0, "adult")
        assert pending_criteria(db, SIG, MODEL) == [("NCT00000001", 0, "adult")]

    def test_excludes_criteria_already_embedded(self, db: duckdb.DuckDBPyConnection) -> None:
        _land_criterion(db, "NCT00000001", 0, "adult")
        write_embeddings(db, [("NCT00000001", 0, _unit(0))], signature=SIG, model=MODEL)

        assert pending_criteria(db, SIG, MODEL) == []

    def test_a_different_embedding_model_re_queues_the_work(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        # Same mechanism as the extraction signature: swapping the model must not
        # leave a table holding vectors from two incomparable spaces.
        _land_criterion(db, "NCT00000001", 0, "adult")
        write_embeddings(db, [("NCT00000001", 0, _unit(0))], signature=SIG, model=MODEL)

        assert pending_criteria(db, SIG, OTHER_MODEL) == [("NCT00000001", 0, "adult")]

    def test_a_different_extraction_signature_is_separate_work(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        _land_criterion(db, "NCT00000001", 0, "adult", signature=SIG)
        _land_criterion(db, "NCT00000001", 0, "adult, revised", signature=OTHER_SIG)
        write_embeddings(db, [("NCT00000001", 0, _unit(0))], signature=SIG, model=MODEL)

        assert pending_criteria(db, OTHER_SIG, MODEL) == [("NCT00000001", 0, "adult, revised")]

    def test_limit_bounds_the_result(self, db: duckdb.DuckDBPyConnection) -> None:
        for i in range(5):
            _land_criterion(db, "NCT00000001", i, f"criterion {i}")
        assert len(pending_criteria(db, SIG, MODEL, limit=2)) == 2


class TestWriteEmbeddings:
    def test_writes_vectors(self, db: duckdb.DuckDBPyConnection) -> None:
        _land_criterion(db, "NCT00000001", 0, "adult")
        assert write_embeddings(db, [("NCT00000001", 0, _unit(3))], signature=SIG, model=MODEL) == 1
        assert gold_stats(db, SIG, MODEL)["vectors"] == 1

    def test_rewriting_updates_rather_than_duplicates(self, db: duckdb.DuckDBPyConnection) -> None:
        _land_criterion(db, "NCT00000001", 0, "adult")
        write_embeddings(db, [("NCT00000001", 0, _unit(0))], signature=SIG, model=MODEL)
        write_embeddings(db, [("NCT00000001", 0, _unit(7))], signature=SIG, model=MODEL)

        assert gold_stats(db, SIG, MODEL)["vectors"] == 1
        row = db.execute(f"SELECT embedding[8] FROM {GOLD_TABLE}").fetchone()  # noqa: S608
        assert row is not None and row[0] == 1.0  # DuckDB arrays are 1-indexed

    def test_an_empty_batch_writes_nothing(self, db: duckdb.DuckDBPyConnection) -> None:
        assert write_embeddings(db, [], signature=SIG, model=MODEL) == 0

    def test_a_wrong_length_vector_is_rejected_before_the_insert(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        # Otherwise a model mismatch surfaces as an opaque insert failure
        # thousands of vectors into a run.
        with pytest.raises(ValueError, match="dimensions"):
            write_embeddings(db, [("NCT00000001", 0, [0.1, 0.2])], signature=SIG, model=MODEL)


class TestSearch:
    def _corpus(self, db: duckdb.DuckDBPyConnection) -> None:
        for nct, ordinal, axis, text in (
            ("NCT00000001", 0, 0, "EGFR exon 19 deletion"),
            ("NCT00000001", 1, 5, "informed consent"),
            ("NCT00000002", 0, 1, "ECOG 0-1"),
            ("NCT00000003", 0, 2, "prior platinum chemotherapy"),
        ):
            _land_criterion(db, nct, ordinal, text)
            write_embeddings(db, [(nct, ordinal, _unit(axis))], signature=SIG, model=MODEL)

    def test_ranks_by_similarity(self, db: duckdb.DuckDBPyConnection) -> None:
        self._corpus(db)
        hits = search_criteria(db, _unit(0), signature=SIG, model=MODEL, k=3)

        assert hits[0][0] == "NCT00000001"
        assert hits[0][1] == pytest.approx(1.0)
        assert all(h[1] == pytest.approx(0.0, abs=1e-6) for h in hits[1:])

    def test_returns_the_criterion_that_matched(self, db: duckdb.DuckDBPyConnection) -> None:
        # The reason is the product here, not the score. A ranking a clinician
        # cannot check is worse than useless.
        self._corpus(db)
        hits = search_criteria(db, _unit(0), signature=SIG, model=MODEL, k=1)

        assert hits[0][2] == 0
        assert hits[0][3] == "EGFR exon 19 deletion"

    def test_a_trial_appears_once_scored_by_its_best_criterion(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        # NCT00000001 has two criteria; a query matching the second must still
        # return the trial exactly once, attributed to that second criterion.
        self._corpus(db)
        hits = search_criteria(db, _unit(5), signature=SIG, model=MODEL, k=10)

        assert [h[0] for h in hits].count("NCT00000001") == 1
        assert hits[0][0] == "NCT00000001"
        assert hits[0][2] == 1

    def test_k_bounds_the_result(self, db: duckdb.DuckDBPyConnection) -> None:
        self._corpus(db)
        assert len(search_criteria(db, _unit(0), signature=SIG, model=MODEL, k=2)) == 2

    def test_another_model_is_not_searched(self, db: duckdb.DuckDBPyConnection) -> None:
        self._corpus(db)
        assert search_criteria(db, _unit(0), signature=SIG, model=OTHER_MODEL) == []

    def test_a_wrong_length_query_is_rejected(self, db: duckdb.DuckDBPyConnection) -> None:
        with pytest.raises(ValueError, match="dimensions"):
            search_criteria(db, [0.1, 0.2], signature=SIG, model=MODEL)


class TestStats:
    def test_counts_vectors_against_criteria(self, db: duckdb.DuckDBPyConnection) -> None:
        _land_criterion(db, "NCT00000001", 0, "a")
        _land_criterion(db, "NCT00000001", 1, "b")
        _land_criterion(db, "NCT00000002", 0, "c")
        write_embeddings(db, [("NCT00000001", 0, _unit(0))], signature=SIG, model=MODEL)

        assert gold_stats(db, SIG, MODEL) == {"vectors": 1, "trials": 1, "criteria": 3}
