"""Bronze storage against a real DuckDB.

Two properties are load-bearing and both caused real incidents:

* **Idempotency.** Landing the same batch twice must update in place. Without it
  an interrupted backfill cannot be resumed by rerunning it, and the platform
  move that cost four minutes would have been a data-migration project.
* **The watermark comes only from completed runs.** Deriving it from
  ``max(bronze.last_update_posted)`` looks equivalent and silently loses data: a
  run that dies at batch 12 of 23 has already landed the newest studies, so the
  next incremental pass starts after them and never fetches the ones that never
  arrived.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import duckdb
import pytest

from scrutatio.clients.ctgov.models import Study
from scrutatio.storage.bronze import (
    BRONZE_TABLE,
    RUNS_TABLE,
    _rows,
    bronze_count,
    bronze_max_update,
    ensure_storage,
    finish_run,
    safe_watermark,
    start_run,
    write_bronze,
)
from scrutatio.storage.db import IN_MEMORY, DatabaseBusyError, connect, database


def _study(nct: str, updated: str | None = "2026-08-14") -> Study:
    status: dict[str, object] = {"overallStatus": "RECRUITING"}
    if updated is not None:
        status["lastUpdatePostDateStruct"] = {"date": updated, "type": "ACTUAL"}
    return Study.model_validate(
        {"protocolSection": {"identificationModule": {"nctId": nct}, "statusModule": status}}
    )


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = connect(IN_MEMORY)
    ensure_storage(conn)
    try:
        yield conn
    finally:
        conn.close()


class TestConnection:
    def test_creates_the_parent_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "deeper" / "scrutatio.duckdb"
        with database(target) as conn:
            conn.execute("CREATE TABLE t (x INT)")
        assert target.exists()

    def test_in_memory_needs_no_path(self) -> None:
        with database(IN_MEMORY) as conn:
            assert conn.execute("SELECT 1").fetchone() == (1,)

    def test_a_locked_file_reports_who_is_holding_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # DuckDB allows one read-write process and refuses every other connection
        # while it is attached — a read-only one included. Measured, not assumed.
        def locked(*_: object, **__: object) -> None:
            raise duckdb.IOException(
                'IO Error: Cannot open file "x.duckdb": <localised OS text>\n\n'
                "File is already open in \nD:\\dev\\python.exe (PID 43812)"
            )

        monkeypatch.setattr(duckdb, "connect", locked)
        with pytest.raises(DatabaseBusyError, match="only one at a time"):
            connect(tmp_path / "x.duckdb")

    def test_other_io_errors_are_not_relabelled_as_a_lock(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The OS half of a lock message is localised, so the marker has to be
        # DuckDB's own wording. A corrupt file must not be reported as "busy".
        def corrupt(*_: object, **__: object) -> None:
            raise duckdb.IOException("IO Error: database file is not valid")

        monkeypatch.setattr(duckdb, "connect", corrupt)
        with pytest.raises(duckdb.IOException, match="not valid"):
            connect(tmp_path / "x.duckdb")


class TestSchemaSetup:
    def test_ddl_is_idempotent(self, db: duckdb.DuckDBPyConnection) -> None:
        ensure_storage(db)
        ensure_storage(db)
        assert bronze_count(db) == 0


class TestRowFlattening:
    def test_duplicate_nct_ids_within_one_batch_are_collapsed(self) -> None:
        # CT.gov paginates over a live index, so a study updated mid-walk can
        # appear on two consecutive pages. ON CONFLICT DO UPDATE refuses to touch
        # the same row twice in one statement, so a repeat would abort the batch.
        rows = _rows([_study("NCT00000001"), _study("NCT00000001"), _study("NCT00000002")])
        assert [r[0] for r in rows] == ["NCT00000001", "NCT00000002"]

    def test_last_occurrence_wins(self) -> None:
        rows = _rows([_study("NCT00000001", "2026-01-01"), _study("NCT00000001", "2026-08-14")])
        assert rows[0][1] == date(2026, 8, 14)

    def test_studies_without_an_nct_id_are_skipped(self) -> None:
        blank = Study.model_validate({"protocolSection": {"statusModule": {}}})
        assert _rows([blank, _study("NCT00000001")]) == _rows([_study("NCT00000001")])

    def test_a_missing_update_date_is_tolerated(self) -> None:
        assert _rows([_study("NCT00000001", updated=None)])[0][1] is None


class TestWriteBronze:
    def test_lands_studies(self, db: duckdb.DuckDBPyConnection) -> None:
        assert write_bronze(db, [_study("NCT00000001"), _study("NCT00000002")]) == 2
        assert bronze_count(db) == 2

    def test_rerunning_the_same_batch_updates_rather_than_duplicates(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        # This is what makes an interrupted backfill safe to simply rerun.
        batch = [_study("NCT00000001"), _study("NCT00000002")]
        write_bronze(db, batch)
        write_bronze(db, batch)

        assert bronze_count(db) == 2

    def test_a_rerun_refreshes_the_payload(self, db: duckdb.DuckDBPyConnection) -> None:
        write_bronze(db, [_study("NCT00000001", "2026-01-01")])
        write_bronze(db, [_study("NCT00000001", "2026-08-14")])

        assert bronze_max_update(db) == date(2026, 8, 14)

    def test_the_raw_payload_is_kept_verbatim(self, db: duckdb.DuckDBPyConnection) -> None:
        # Silver is rebuilt from Bronze rather than re-fetched; the registry is a
        # moving target, so a re-fetch is not a rerun.
        write_bronze(db, [_study("NCT00000001")])

        row = db.execute(f"SELECT raw FROM {BRONZE_TABLE}").fetchone()  # noqa: S608
        assert row is not None
        payload = json.loads(row[0])
        assert payload["protocolSection"]["identificationModule"]["nctId"] == "NCT00000001"

    def test_an_empty_batch_writes_nothing(self, db: duckdb.DuckDBPyConnection) -> None:
        assert write_bronze(db, []) == 0
        assert bronze_count(db) == 0

    def test_duplicates_within_a_batch_do_not_abort_it(self, db: duckdb.DuckDBPyConnection) -> None:
        assert write_bronze(db, [_study("NCT00000001"), _study("NCT00000001")]) == 1
        assert bronze_count(db) == 1


class TestWatermark:
    def test_no_completed_run_means_no_resume_point(self, db: duckdb.DuckDBPyConnection) -> None:
        # Forcing a full pass is correct; resuming from a point never reached
        # would skip everything that never landed.
        assert safe_watermark(db) is None

    def test_an_incomplete_run_publishes_nothing(self, db: duckdb.DuckDBPyConnection) -> None:
        write_bronze(db, [_study("NCT00000001", "2026-08-14")])
        start_run(db, "run-1", "full")
        finish_run(db, "run-1", written=1, complete=False)

        assert safe_watermark(db) is None
        # …even though Bronze itself already knows about the newer study.
        assert bronze_max_update(db) == date(2026, 8, 14)

    def test_a_completed_run_publishes_the_bronze_maximum(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        write_bronze(db, [_study("NCT00000001", "2026-08-14")])
        start_run(db, "run-1", "full")
        finish_run(db, "run-1", written=1, complete=True)

        assert safe_watermark(db) == date(2026, 8, 14)

    def test_a_later_incomplete_run_does_not_override_a_good_watermark(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        write_bronze(db, [_study("NCT00000001", "2026-08-01")])
        start_run(db, "run-1", "full")
        finish_run(db, "run-1", written=1, complete=True)

        write_bronze(db, [_study("NCT00000002", "2026-08-14")])
        start_run(db, "run-2", "incremental")
        finish_run(db, "run-2", written=1, complete=False)

        assert safe_watermark(db) == date(2026, 8, 1)

    def test_the_newest_completed_run_wins(self, db: duckdb.DuckDBPyConnection) -> None:
        write_bronze(db, [_study("NCT00000001", "2026-08-01")])
        start_run(db, "run-1", "full")
        finish_run(db, "run-1", written=1, complete=True)

        write_bronze(db, [_study("NCT00000002", "2026-08-14")])
        start_run(db, "run-2", "incremental")
        finish_run(db, "run-2", written=1, complete=True)

        assert safe_watermark(db) == date(2026, 8, 14)


class TestRunLedger:
    def test_a_started_run_is_recorded_as_incomplete(self, db: duckdb.DuckDBPyConnection) -> None:
        start_run(db, "run-1", "full")

        row = db.execute(
            f"SELECT mode, complete, finished_at FROM {RUNS_TABLE} WHERE run_id = ?",  # noqa: S608
            ["run-1"],
        ).fetchone()
        assert row is not None
        assert row[0] == "full"
        assert row[1] is False
        assert row[2] is None

    def test_finishing_records_the_written_count(self, db: duckdb.DuckDBPyConnection) -> None:
        start_run(db, "run-1", "full")
        finish_run(db, "run-1", written=42, complete=True)

        row = db.execute(
            f"SELECT studies_written, finished_at FROM {RUNS_TABLE} WHERE run_id = ?",  # noqa: S608
            ["run-1"],
        ).fetchone()
        assert row is not None
        assert row[0] == 42
        assert row[1] is not None


class TestHostileInput:
    def test_a_run_id_carrying_sql_is_bound_not_interpolated(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        # The old COPY INTO path interpolated this into a path literal and needed
        # a charset guard. Binding removes the class of problem entirely.
        hostile = "run'; DROP TABLE bronze_studies; --"
        start_run(db, hostile, "full")
        finish_run(db, hostile, written=0, complete=False)

        assert bronze_count(db) == 0  # the table is still there to be counted
        row = db.execute(
            f"SELECT count(*) FROM {RUNS_TABLE} WHERE run_id = ?",  # noqa: S608
            [hostile],
        ).fetchone()
        assert row is not None and row[0] == 1
