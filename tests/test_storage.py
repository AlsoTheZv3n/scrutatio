"""Storage tests.

The load-bearing property is idempotency: an interrupted backfill gets rerun,
and a rerun must update rather than duplicate. That is a MERGE, never an
INSERT, and the test asserts on the statement actually sent.
"""

from __future__ import annotations

import gzip
import json
from typing import Any

import httpx
import pytest
import respx

from scrutatio.config import Settings
from scrutatio.ctgov.models import Study
from scrutatio.storage import BRONZE_TABLE, SqlClient, SqlError, ensure_storage, write_bronze
from scrutatio.storage.bronze import (
    UnsafeBatchNameError,
    _to_ndjson,
    bronze_count,
    bronze_max_update,
    finish_run,
    safe_watermark,
)

HOST = "https://dbc-00000000-0000.cloud.databricks.com"
TOKEN = "fake-token-for-tests-only"
STATEMENTS_URL = f"{HOST}/api/2.0/sql/statements"
WAREHOUSES_URL = f"{HOST}/api/2.0/sql/warehouses"


def _study(nct: str, updated: str | None = "2026-08-14") -> Study:
    status: dict[str, Any] = {"overallStatus": "RECRUITING"}
    if updated:
        status["lastUpdatePostDateStruct"] = {"date": updated, "type": "ACTUAL"}
    return Study.model_validate(
        {
            "protocolSection": {
                "identificationModule": {"nctId": nct},
                "statusModule": status,
                "eligibilityModule": {"eligibilityCriteria": "INCLUSION: adult."},
            }
        }
    )


def _ok(rows: list[list[Any]] | None = None) -> httpx.Response:
    body: dict[str, Any] = {"statement_id": "s1", "status": {"state": "SUCCEEDED"}}
    if rows is not None:
        body["result"] = {"data_array": rows}
    return httpx.Response(200, json=body)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, databricks_host=HOST, databricks_token=TOKEN)  # type: ignore[call-arg]


@pytest.fixture
def sql(settings: Settings) -> SqlClient:
    return SqlClient(settings, warehouse_id="wh1")


class TestSqlClient:
    def test_refuses_without_credentials(self) -> None:
        with pytest.raises(ValueError, match="DATABRICKS_HOST"):
            SqlClient(Settings(_env_file=None))  # type: ignore[call-arg]

    @respx.mock
    def test_discovers_the_first_warehouse(self, settings: Settings) -> None:
        respx.get(WAREHOUSES_URL).mock(
            httpx.Response(200, json={"warehouses": [{"id": "found"}, {"id": "other"}]})
        )
        with SqlClient(settings) as client:
            assert client.warehouse_id() == "found"

    @respx.mock
    def test_errors_when_no_warehouse_exists(self, settings: Settings) -> None:
        respx.get(WAREHOUSES_URL).mock(httpx.Response(200, json={"warehouses": []}))
        with SqlClient(settings) as client, pytest.raises(SqlError, match="No SQL warehouse"):
            client.warehouse_id()

    @respx.mock
    def test_returns_rows(self, sql: SqlClient) -> None:
        respx.post(STATEMENTS_URL).mock(_ok([["a", 1]]))
        assert sql.execute("SELECT 1") == [["a", 1]]

    @respx.mock
    def test_failed_statement_surfaces_the_message(self, sql: SqlClient) -> None:
        respx.post(STATEMENTS_URL).mock(
            httpx.Response(
                200,
                json={
                    "statement_id": "s1",
                    "status": {"state": "FAILED", "error": {"message": "DELTA_FAILED_TO_MERGE"}},
                },
            )
        )
        with pytest.raises(SqlError, match="DELTA_FAILED_TO_MERGE"):
            sql.execute("MERGE ...")

    @respx.mock
    def test_polls_until_terminal(self, sql: SqlClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("scrutatio.storage.sql.time.sleep", lambda _: None)
        respx.post(STATEMENTS_URL).mock(
            httpx.Response(200, json={"statement_id": "s1", "status": {"state": "RUNNING"}})
        )
        respx.get(f"{STATEMENTS_URL}/s1").mock(_ok([["done"]]))
        assert sql.execute("SELECT 1") == [["done"]]

    @respx.mock
    def test_rejected_statement_raises(self, sql: SqlClient) -> None:
        respx.post(STATEMENTS_URL).mock(httpx.Response(400, text="syntax error"))
        with pytest.raises(SqlError, match="rejected"):
            sql.execute("SELEKT 1")

    @respx.mock
    def test_upload_targets_the_volumes_path(self, sql: SqlClient) -> None:
        route = respx.put(
            f"{HOST}/api/2.0/fs/files/Volumes/workspace/scrutatio/landing/b.ndjson.gz"
        ).mock(httpx.Response(204))
        sql.upload_file("workspace/scrutatio/landing/b.ndjson.gz", b"data")
        assert route.calls[0].request.url.params["overwrite"] == "true"

    @respx.mock
    def test_failed_upload_raises(self, sql: SqlClient) -> None:
        respx.put(url__startswith=f"{HOST}/api/2.0/fs/files/").mock(httpx.Response(403))
        with pytest.raises(SqlError, match="Upload"):
            sql.upload_file("workspace/scrutatio/landing/b.gz", b"data")


class TestSerialisation:
    def test_ndjson_is_one_line_per_study(self) -> None:
        payload, rows = _to_ndjson([_study("NCT00000001"), _study("NCT00000002")])
        lines = gzip.decompress(payload).decode().strip().split("\n")
        assert rows == 2
        assert len(lines) == 2
        assert json.loads(lines[0])["nct_id"] == "NCT00000001"

    def test_raw_payload_is_preserved(self) -> None:
        payload, _ = _to_ndjson([_study("NCT00000001")])
        record = json.loads(gzip.decompress(payload).decode().strip())
        assert "eligibilityCriteria" in record["raw"]

    def test_missing_date_becomes_null(self) -> None:
        payload, _ = _to_ndjson([_study("NCT00000001", updated=None)])
        record = json.loads(gzip.decompress(payload).decode().strip())
        assert record["last_update_posted"] is None

    def test_study_without_nct_id_is_skipped(self) -> None:
        _, rows = _to_ndjson([Study.model_validate({})])
        assert rows == 0


class TestWriteBronze:
    @respx.mock
    def test_empty_batch_touches_nothing(self, sql: SqlClient) -> None:
        route = respx.post(STATEMENTS_URL).mock(_ok())
        assert write_bronze(sql, [], batch="b0") == 0
        assert route.call_count == 0

    @respx.mock
    def test_merges_rather_than_inserts(self, sql: SqlClient) -> None:
        # An INSERT here would duplicate every study on the next incremental run.
        respx.put(url__startswith=f"{HOST}/api/2.0/fs/files/").mock(httpx.Response(204))
        route = respx.post(STATEMENTS_URL).mock(_ok())

        assert write_bronze(sql, [_study("NCT00000001")], batch="b0") == 1

        statements = [json.loads(c.request.content)["statement"] for c in route.calls]
        merge = next(s for s in statements if "MERGE INTO" in s)
        assert BRONZE_TABLE in merge
        assert "ON target.nct_id = source.nct_id" in merge
        assert not any(s.strip().startswith("INSERT INTO") for s in statements)

    @respx.mock
    def test_dates_are_cast_defensively(self, sql: SqlClient) -> None:
        # try_cast, not cast: one malformed date must not abort the batch.
        respx.put(url__startswith=f"{HOST}/api/2.0/fs/files/").mock(httpx.Response(204))
        route = respx.post(STATEMENTS_URL).mock(_ok())

        write_bronze(sql, [_study("NCT00000001")], batch="b0")

        merge = next(
            json.loads(c.request.content)["statement"]
            for c in route.calls
            if "MERGE INTO" in json.loads(c.request.content)["statement"]
        )
        assert merge.count("try_cast") == 2  # update branch and insert branch

    @respx.mock
    def test_batch_name_determines_the_upload_path(self, sql: SqlClient) -> None:
        route = respx.put(url__startswith=f"{HOST}/api/2.0/fs/files/").mock(httpx.Response(204))
        respx.post(STATEMENTS_URL).mock(_ok())

        write_bronze(sql, [_study("NCT00000001")], batch="backfill-042")

        assert "backfill-042.ndjson.gz" in str(route.calls[0].request.url)


class TestBatchNameSafety:
    """The batch name is interpolated into a SQL path literal that the
    Statement Execution API cannot parameterise. A quote in the name would
    break out of the literal, so the charset is restricted at the door.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "b'; DROP TABLE workspace.scrutatio.bronze_studies; --",
            "batch' OR '1'='1",
            "../../etc/passwd",
            "with space",
            "",
            "x" * 65,
        ],
    )
    def test_dangerous_names_are_rejected(self, sql: SqlClient, name: str) -> None:
        with pytest.raises(UnsafeBatchNameError):
            write_bronze(sql, [_study("NCT00000001")], batch=name)

    @respx.mock
    def test_rejection_happens_before_any_request(self, sql: SqlClient) -> None:
        route = respx.post(STATEMENTS_URL).mock(_ok())
        upload = respx.put(url__startswith=f"{HOST}/api/2.0/fs/files/").mock(httpx.Response(204))

        with pytest.raises(UnsafeBatchNameError):
            write_bronze(sql, [_study("NCT00000001")], batch="bad'; DROP TABLE x; --")

        assert route.call_count == 0
        assert upload.call_count == 0

    @pytest.mark.parametrize("name", ["backfill-000", "batch_1", "inc.2026-08-15", "a"])
    def test_ordinary_names_are_accepted(self, sql: SqlClient, name: str) -> None:
        with respx.mock:
            respx.put(url__startswith=f"{HOST}/api/2.0/fs/files/").mock(httpx.Response(204))
            respx.post(STATEMENTS_URL).mock(_ok())
            assert write_bronze(sql, [_study("NCT00000001")], batch=name) == 1


class TestSchemaSetup:
    @respx.mock
    def test_ddl_is_idempotent_and_enables_cdf(self, sql: SqlClient) -> None:
        # Change Data Feed must be on at creation; retrofitting it is painful.
        route = respx.post(STATEMENTS_URL).mock(_ok())
        ensure_storage(sql)

        statements = [json.loads(c.request.content)["statement"] for c in route.calls]
        assert all("IF NOT EXISTS" in s for s in statements)
        assert any("enableChangeDataFeed = true" in s for s in statements)


class TestQueries:
    @respx.mock
    def test_count(self, sql: SqlClient) -> None:
        respx.post(STATEMENTS_URL).mock(_ok([[200]]))
        assert bronze_count(sql) == 200

    @respx.mock
    def test_count_on_empty_result(self, sql: SqlClient) -> None:
        respx.post(STATEMENTS_URL).mock(_ok([]))
        assert bronze_count(sql) == 0

    @respx.mock
    def test_max_update_is_diagnostic_only(self, sql: SqlClient) -> None:
        respx.post(STATEMENTS_URL).mock(_ok([["2026-08-14"]]))
        assert bronze_max_update(sql) == "2026-08-14"

    @respx.mock
    def test_max_update_is_none_on_empty_table(self, sql: SqlClient) -> None:
        respx.post(STATEMENTS_URL).mock(_ok([[None]]))
        assert bronze_max_update(sql) is None


class TestWatermarkSafety:
    """A run that dies at batch 12 of 23 still leaves the newest studies it did
    land. Deriving the resume point from max(Bronze) would start the next pass
    after them and permanently skip everything that never arrived.
    """

    @respx.mock
    def test_resume_point_comes_from_completed_runs_only(self, sql: SqlClient) -> None:
        route = respx.post(STATEMENTS_URL).mock(_ok([["2026-08-01"]]))
        assert safe_watermark(sql) == "2026-08-01"

        statement = json.loads(route.calls[0].request.content)["statement"]
        assert "bronze_runs" in statement
        assert "WHERE complete" in statement
        assert "bronze_studies" not in statement

    @respx.mock
    def test_no_completed_run_forces_a_full_pass(self, sql: SqlClient) -> None:
        respx.post(STATEMENTS_URL).mock(_ok([]))
        assert safe_watermark(sql) is None

    @respx.mock
    def test_incomplete_run_publishes_no_watermark(self, sql: SqlClient) -> None:
        route = respx.post(STATEMENTS_URL).mock(_ok())
        finish_run(sql, "run-1", written=6000, complete=False)

        statement = json.loads(route.calls[0].request.content)["statement"]
        assert "watermark = NULL" in statement
        assert "complete = false" in statement

    @respx.mock
    def test_complete_run_publishes_the_watermark(self, sql: SqlClient) -> None:
        route = respx.post(STATEMENTS_URL).mock(_ok())
        finish_run(sql, "run-1", written=11200, complete=True)

        statement = json.loads(route.calls[0].request.content)["statement"]
        assert "max(last_update_posted)" in statement
        assert "complete = true" in statement

    def test_run_id_is_charset_checked(self, sql: SqlClient) -> None:
        with pytest.raises(UnsafeBatchNameError):
            finish_run(sql, "run'; DROP TABLE x; --", written=1, complete=True)


class TestDeduplication:
    """CT.gov paginates over a live index, so a study updated mid-walk can appear
    on two consecutive pages. Delta's MERGE aborts with
    DELTA_MULTIPLE_SOURCE_ROW_MATCHING_TARGET_ROW on a repeated source key, or
    silently inserts both when the key is new.
    """

    def test_duplicate_nct_ids_collapse_to_one_row(self) -> None:
        payload, rows = _to_ndjson([_study("NCT00000001"), _study("NCT00000001")])
        assert rows == 1
        assert len(gzip.decompress(payload).decode().strip().split("\n")) == 1

    def test_last_occurrence_wins(self) -> None:
        stale = _study("NCT00000001", updated="2026-01-01")
        fresh = _study("NCT00000001", updated="2026-08-14")
        payload, _ = _to_ndjson([stale, fresh])
        record = json.loads(gzip.decompress(payload).decode().strip())
        assert record["last_update_posted"] == "2026-08-14"

    def test_distinct_studies_are_all_kept(self) -> None:
        _, rows = _to_ndjson([_study(f"NCT{i:08d}") for i in range(1, 6)])
        assert rows == 5
