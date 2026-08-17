"""Silver tests.

Two properties carry the design: an extraction must be attributable to what
produced it (the signature), and re-extracting a trial must replace its rows
rather than merge with them. Everything else follows from those.

These run against a real in-memory DuckDB. The previous versions asserted on the
text of generated SQL statements, which tested a recording of the storage layer
rather than the storage layer; here the assertions are about what ends up in the
tables.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import duckdb
import pytest

from scrutatio.config import Settings
from scrutatio.extraction.runner import ExtractionOutcome
from scrutatio.extraction.schema import Criterion, ExtractedEligibility
from scrutatio.storage.bronze import BRONZE_TABLE, ensure_storage
from scrutatio.storage.db import IN_MEMORY, connect
from scrutatio.storage.silver import (
    FAILURES_TABLE,
    SILVER_TABLE,
    ensure_silver,
    extraction_signature,
    pending_trials,
    silver_stats,
    write_silver,
)

SIG = "abc123def4567890"
OTHER_SIG = "0000111122223333"


def _outcome(nct: str, *, kinds: list[str] | None = None) -> ExtractionOutcome:
    criteria = [
        Criterion(criterion_id=f"c-{i}", text=f"criterion {i}", kind=k, is_exclusion=(i % 2 == 1))
        for i, k in enumerate(kinds or ["condition", "biomarker"])
    ]
    return ExtractionOutcome(nct_id=nct, result=ExtractedEligibility(criteria=criteria))


def _failure(nct: str, error: str = "HTTP 429") -> ExtractionOutcome:
    return ExtractionOutcome(nct_id=nct, result=None, error=error)


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = connect(IN_MEMORY)
    ensure_storage(conn)
    ensure_silver(conn)
    try:
        yield conn
    finally:
        conn.close()


def _land_bronze(db: duckdb.DuckDBPyConnection, nct: str, eligibility: str | None) -> None:
    """Put one study into Bronze with the given eligibility prose."""
    module = {} if eligibility is None else {"eligibilityCriteria": eligibility}
    raw = json.dumps({"protocolSection": {"eligibilityModule": module}})
    db.execute(
        f"INSERT INTO {BRONZE_TABLE} VALUES (?, NULL, ?, current_timestamp)",  # noqa: S608
        [nct, raw],
    )


def _silver_rows(db: duckdb.DuckDBPyConnection, signature: str = SIG) -> list[tuple]:
    return db.execute(
        f"SELECT nct_id, ordinal, kind, is_exclusion FROM {SILVER_TABLE} "  # noqa: S608
        "WHERE signature = ? ORDER BY nct_id, ordinal",
        [signature],
    ).fetchall()


class TestExtractionSignature:
    """Rows must be attributable to what produced them. The 9-to-20 taxonomy
    change silently invalidated every earlier extraction; nothing in the data
    said so. The signature makes that visible and self-correcting.
    """

    def test_is_stable_for_identical_configuration(self) -> None:
        assert extraction_signature() == extraction_signature()

    def test_changes_when_the_model_changes(self) -> None:
        a = extraction_signature(Settings(_env_file=None))  # type: ignore[call-arg]
        b = extraction_signature(
            Settings(_env_file=None, extraction_model="openai/gpt-5-nano")  # type: ignore[call-arg]
        )
        assert a != b

    def test_is_short_enough_to_read(self) -> None:
        assert len(extraction_signature()) == 16


class TestPendingTrials:
    def test_returns_trials_bronze_has_and_silver_does_not(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        _land_bronze(db, "NCT00000001", "INCLUSION: adult.")
        assert pending_trials(db, SIG) == [("NCT00000001", "INCLUSION: adult.")]

    def test_excludes_trials_already_extracted_under_this_signature(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        _land_bronze(db, "NCT00000001", "text")
        write_silver(db, [_outcome("NCT00000001")], signature=SIG)

        assert pending_trials(db, SIG) == []

    def test_a_different_signature_makes_the_trial_pending_again(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        # This is the taxonomy-change behaviour: changing the model, prompt or
        # schema re-queues exactly the affected work, with no manual bookkeeping.
        _land_bronze(db, "NCT00000001", "text")
        write_silver(db, [_outcome("NCT00000001")], signature=SIG)

        assert pending_trials(db, OTHER_SIG) == [("NCT00000001", "text")]

    def test_excludes_trials_that_exhausted_their_attempts(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        _land_bronze(db, "NCT00000001", "text")
        for _ in range(3):
            write_silver(db, [_failure("NCT00000001", "schema-invalid")], signature=SIG)

        assert pending_trials(db, SIG, max_attempts=3) == []

    def test_retry_failed_includes_them_again(self, db: duckdb.DuckDBPyConnection) -> None:
        _land_bronze(db, "NCT00000001", "text")
        for _ in range(3):
            write_silver(db, [_failure("NCT00000001", "schema-invalid")], signature=SIG)

        assert pending_trials(db, SIG, max_attempts=3, retry_failed=True) == [
            ("NCT00000001", "text")
        ]

    def test_trials_without_eligibility_text_are_dropped(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        _land_bronze(db, "NCT00000001", None)
        _land_bronze(db, "NCT00000002", "text")

        assert pending_trials(db, SIG) == [("NCT00000002", "text")]

    def test_limit_bounds_the_result(self, db: duckdb.DuckDBPyConnection) -> None:
        for i in range(5):
            _land_bronze(db, f"NCT0000000{i}", "text")

        assert len(pending_trials(db, SIG, limit=2)) == 2


class TestWriteSilver:
    def test_writes_one_row_per_criterion(self, db: duckdb.DuckDBPyConnection) -> None:
        rows, trials = write_silver(
            db, [_outcome("NCT00000001", kinds=["condition", "lab", "ecog"])], signature=SIG
        )

        assert (rows, trials) == (3, 1)
        assert len(_silver_rows(db)) == 3

    def test_ordinal_preserves_criterion_order(self, db: duckdb.DuckDBPyConnection) -> None:
        write_silver(
            db, [_outcome("NCT00000001", kinds=["condition", "lab", "washout"])], signature=SIG
        )

        assert [(r[1], r[2]) for r in _silver_rows(db)] == [
            (0, "condition"),
            (1, "lab"),
            (2, "washout"),
        ]

    def test_replacing_a_trial_leaves_no_orphan_rows(self, db: duckdb.DuckDBPyConnection) -> None:
        # Criterion ids are model-generated and unstable across runs, so merging
        # would leave rows from the previous extraction's shape behind.
        write_silver(
            db, [_outcome("NCT00000001", kinds=["condition", "lab", "ecog"])], signature=SIG
        )
        write_silver(db, [_outcome("NCT00000001", kinds=["biomarker"])], signature=SIG)

        assert [r[2] for r in _silver_rows(db)] == ["biomarker"]

    def test_replacement_does_not_touch_another_signature(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        # Without the signature filter, re-extracting under a new taxonomy would
        # wipe the previous signature's rows too.
        write_silver(db, [_outcome("NCT00000001", kinds=["condition", "lab"])], signature=OTHER_SIG)
        write_silver(db, [_outcome("NCT00000001", kinds=["biomarker"])], signature=SIG)

        assert len(_silver_rows(db, OTHER_SIG)) == 2
        assert len(_silver_rows(db, SIG)) == 1

    def test_failed_outcomes_write_no_criteria(self, db: duckdb.DuckDBPyConnection) -> None:
        assert write_silver(db, [_failure("NCT00000001")], signature=SIG) == (0, 0)
        assert _silver_rows(db) == []


class TestFailureAccounting:
    def _attempts(self, db: duckdb.DuckDBPyConnection, nct: str) -> int | None:
        row = db.execute(
            f"SELECT attempts FROM {FAILURES_TABLE} WHERE nct_id = ? AND signature = ?",  # noqa: S608
            [nct, SIG],
        ).fetchone()
        return row[0] if row else None

    def test_genuine_failures_increment_an_attempt_counter(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        for _ in range(2):
            write_silver(db, [_failure("NCT00000001", "schema-invalid JSON")], signature=SIG)

        assert self._attempts(db, "NCT00000001") == 2

    def test_throttled_trials_are_left_pending_not_retired(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        # A 429 describes the moment, not the trial. Counting it against the
        # attempt budget permanently removes a trial nothing is wrong with —
        # measured at three exclusions in the first minute of a real run.
        write_silver(
            db,
            [_failure("NCT00000001", "Request failed after 6 attempts: HTTP 429")],
            signature=SIG,
        )

        assert self._attempts(db, "NCT00000001") is None

    def test_a_mixed_batch_records_only_the_genuine_failure(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        write_silver(
            db,
            [
                _outcome("NCT00000001"),
                _failure("NCT00000002", "HTTP 429"),
                _failure("NCT00000003", "schema-invalid JSON"),
            ],
            signature=SIG,
        )

        assert self._attempts(db, "NCT00000002") is None
        assert self._attempts(db, "NCT00000003") == 1

    def test_a_trial_that_now_succeeds_stops_counting_as_failed(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        write_silver(db, [_failure("NCT00000001", "schema-invalid")], signature=SIG)
        write_silver(db, [_outcome("NCT00000001")], signature=SIG)

        assert self._attempts(db, "NCT00000001") is None

    def test_error_text_is_truncated(self, db: duckdb.DuckDBPyConnection) -> None:
        write_silver(db, [_failure("NCT00000001", "e" * 5000)], signature=SIG)

        row = db.execute(
            f"SELECT error FROM {FAILURES_TABLE} WHERE nct_id = ?",  # noqa: S608
            ["NCT00000001"],
        ).fetchone()
        assert row is not None
        assert len(row[0]) == 400


class TestHostileInput:
    """Error text and criterion prose arrive from an HTTP layer and reach SQL.

    Values are bound rather than interpolated, so this is now a property of the
    driver rather than of a quoting helper — but it is exactly the property that
    a quoting helper used to be responsible for, so it stays tested.
    """

    def test_sql_in_an_error_message_is_stored_verbatim(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        hostile = "x'; DROP TABLE silver_criteria; --"
        write_silver(db, [_failure("NCT00000001", hostile)], signature=SIG)

        row = db.execute(
            f"SELECT error FROM {FAILURES_TABLE} WHERE nct_id = ?",  # noqa: S608
            ["NCT00000001"],
        ).fetchone()
        assert row is not None and row[0] == hostile
        # The table it tried to drop is still there.
        assert db.execute(f"SELECT count(*) FROM {SILVER_TABLE}").fetchone() is not None  # noqa: S608

    def test_sql_in_criterion_text_is_stored_verbatim(self, db: duckdb.DuckDBPyConnection) -> None:
        hostile = "Patient must not have '); DELETE FROM bronze_studies; --"
        outcome = ExtractionOutcome(
            nct_id="NCT00000001",
            result=ExtractedEligibility(
                criteria=[
                    Criterion(criterion_id="c-0", text=hostile, kind="other", is_exclusion=True)
                ]
            ),
        )
        write_silver(db, [outcome], signature=SIG)

        row = db.execute(f"SELECT text FROM {SILVER_TABLE}").fetchone()  # noqa: S608
        assert row is not None and row[0] == hostile


class TestStats:
    def test_reports_progress_counters(self, db: duckdb.DuckDBPyConnection) -> None:
        _land_bronze(db, "NCT00000001", "text")
        _land_bronze(db, "NCT00000002", "text")
        _land_bronze(db, "NCT00000003", None)
        write_silver(
            db, [_outcome("NCT00000001", kinds=["condition", "lab", "ecog"])], signature=SIG
        )
        write_silver(db, [_failure("NCT00000002", "schema-invalid")], signature=SIG)

        assert silver_stats(db, SIG) == {
            "trials": 1,
            "criteria": 3,
            "failed": 1,
            # NCT00000003 has no eligibility text, so it is not in scope at all.
            "total": 2,
        }

    def test_an_empty_database_is_all_zero(self, db: duckdb.DuckDBPyConnection) -> None:
        assert silver_stats(db, SIG) == {"trials": 0, "criteria": 0, "failed": 0, "total": 0}


class TestSchemaSetup:
    def test_ddl_is_idempotent(self, db: duckdb.DuckDBPyConnection) -> None:
        ensure_silver(db)
        ensure_silver(db)
        assert silver_stats(db, SIG)["criteria"] == 0
