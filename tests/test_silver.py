"""Silver tests.

Two properties carry the design: an extraction must be attributable to what
produced it (the signature), and re-extracting a trial must replace its rows
rather than merge with them. Everything else follows from those.
"""

from __future__ import annotations

import gzip
import json
from typing import Any

import httpx
import pytest
import respx

from scrutatio.config import Settings
from scrutatio.extraction.runner import ExtractionOutcome
from scrutatio.extraction.schema import Criterion, ExtractedEligibility
from scrutatio.storage.silver import (
    FAILURES_TABLE,
    SILVER_TABLE,
    _quote_list,
    _sql_string,
    _to_ndjson,
    ensure_silver,
    extraction_signature,
    pending_trials,
    silver_stats,
    write_silver,
)
from scrutatio.storage.sql import SqlClient

HOST = "https://dbc-00000000-0000.cloud.databricks.com"
TOKEN = "fake-token-for-tests-only"
STATEMENTS_URL = f"{HOST}/api/2.0/sql/statements"
SIG = "abc123def4567890"


def _ok(rows: list[list[Any]] | None = None) -> httpx.Response:
    body: dict[str, Any] = {"statement_id": "s1", "status": {"state": "SUCCEEDED"}}
    if rows is not None:
        body["result"] = {"data_array": rows}
    return httpx.Response(200, json=body)


def _outcome(nct: str, *, kinds: list[str] | None = None) -> ExtractionOutcome:
    criteria = [
        Criterion(criterion_id=f"c-{i}", text=f"criterion {i}", kind=k, is_exclusion=(i % 2 == 1))
        for i, k in enumerate(kinds or ["condition", "biomarker"])
    ]
    return ExtractionOutcome(nct_id=nct, result=ExtractedEligibility(criteria=criteria))


def _failure(nct: str, error: str = "HTTP 429") -> ExtractionOutcome:
    return ExtractionOutcome(nct_id=nct, result=None, error=error)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, databricks_host=HOST, databricks_token=TOKEN)  # type: ignore[call-arg]


@pytest.fixture
def sql(settings: Settings) -> SqlClient:
    return SqlClient(settings, warehouse_id="wh1")


def _statements(route: Any) -> list[str]:
    return [json.loads(c.request.content)["statement"] for c in route.calls]


class TestExtractionSignature:
    """Rows must be attributable to what produced them. The 9-to-20 taxonomy
    change silently invalidated every earlier extraction; nothing in the data
    said so. The signature makes that visible and self-correcting.
    """

    def test_is_stable_for_identical_configuration(self) -> None:
        assert extraction_signature() == extraction_signature()

    def test_changes_when_the_model_endpoint_changes(self) -> None:
        a = extraction_signature(Settings(_env_file=None))  # type: ignore[call-arg]
        b = extraction_signature(
            Settings(_env_file=None, chat_endpoint="databricks-claude-opus-5")  # type: ignore[call-arg]
        )
        assert a != b

    def test_is_short_enough_to_read(self) -> None:
        assert len(extraction_signature()) == 16


class TestSerialisation:
    def test_numbers_and_booleans_are_written_as_strings(self) -> None:
        # COPY INTO infers a bare 0 as bigint, which fails the STRING staging
        # column with DELTA_FAILED_TO_MERGE_FIELDS. This exact bug aborted a run.
        payload, _, _ = _to_ndjson([_outcome("NCT00000001")], SIG)
        record = json.loads(gzip.decompress(payload).decode().strip().split("\n")[0])

        assert record["ordinal"] == "0"
        assert isinstance(record["ordinal"], str)
        assert record["is_exclusion"] in {"true", "false"}
        assert isinstance(record["is_exclusion"], str)

    def test_one_line_per_criterion(self) -> None:
        payload, rows, trials = _to_ndjson(
            [_outcome("NCT00000001", kinds=["condition", "lab", "ecog"])], SIG
        )
        assert (rows, trials) == (3, 1)
        assert len(gzip.decompress(payload).decode().strip().split("\n")) == 3

    def test_ordinal_preserves_criterion_order(self) -> None:
        payload, _, _ = _to_ndjson(
            [_outcome("NCT00000001", kinds=["condition", "lab", "washout"])], SIG
        )
        lines = [json.loads(x) for x in gzip.decompress(payload).decode().strip().split("\n")]
        assert [x["ordinal"] for x in lines] == ["0", "1", "2"]

    def test_failed_outcomes_contribute_nothing(self) -> None:
        _, rows, trials = _to_ndjson([_failure("NCT00000001")], SIG)
        assert (rows, trials) == (0, 0)

    def test_signature_is_stamped_on_every_row(self) -> None:
        payload, _, _ = _to_ndjson([_outcome("NCT00000001")], SIG)
        lines = [json.loads(x) for x in gzip.decompress(payload).decode().strip().split("\n")]
        assert all(x["signature"] == SIG for x in lines)


class TestPendingTrials:
    @respx.mock
    def test_excludes_trials_already_extracted_under_this_signature(self, sql: SqlClient) -> None:
        route = respx.post(STATEMENTS_URL).mock(_ok([["NCT00000001", "INCLUSION: adult."]]))

        result = pending_trials(sql, SIG, limit=10)

        assert result == [("NCT00000001", "INCLUSION: adult.")]
        statement = _statements(route)[0]
        assert "NOT EXISTS" in statement
        assert SILVER_TABLE in statement
        assert f"s.signature = '{SIG}'" in statement

    @respx.mock
    def test_excludes_trials_that_exhausted_their_attempts(self, sql: SqlClient) -> None:
        route = respx.post(STATEMENTS_URL).mock(_ok([]))
        pending_trials(sql, SIG, max_attempts=3)

        statement = _statements(route)[0]
        assert FAILURES_TABLE in statement
        assert "f.attempts >= 3" in statement

    @respx.mock
    def test_retry_failed_drops_the_failure_filter(self, sql: SqlClient) -> None:
        route = respx.post(STATEMENTS_URL).mock(_ok([]))
        pending_trials(sql, SIG, retry_failed=True)

        assert FAILURES_TABLE not in _statements(route)[0]

    @respx.mock
    def test_rows_without_eligibility_text_are_dropped(self, sql: SqlClient) -> None:
        respx.post(STATEMENTS_URL).mock(_ok([["NCT00000001", None], ["NCT00000002", "text"]]))
        assert pending_trials(sql, SIG) == [("NCT00000002", "text")]

    @respx.mock
    def test_limit_is_int_cast_into_the_statement(self, sql: SqlClient) -> None:
        route = respx.post(STATEMENTS_URL).mock(_ok([]))
        pending_trials(sql, SIG, limit=25)
        assert "LIMIT 25" in _statements(route)[0]


class TestWriteSilver:
    @respx.mock
    def test_replaces_rather_than_merges(self, sql: SqlClient) -> None:
        # Criterion ids are model-generated and unstable across runs, so a MERGE
        # would leave orphaned rows from the previous extraction's shape.
        respx.put(url__startswith=f"{HOST}/api/2.0/fs/files/").mock(httpx.Response(204))
        route = respx.post(STATEMENTS_URL).mock(_ok())

        rows, trials = write_silver(sql, [_outcome("NCT00000001")], batch="b0", signature=SIG)

        assert (rows, trials) == (2, 1)
        statements = _statements(route)
        assert any(s.strip().startswith("DELETE FROM") and SILVER_TABLE in s for s in statements)
        assert any("INSERT INTO" in s for s in statements)
        assert not any("MERGE INTO" in s and SILVER_TABLE in s for s in statements)

    @respx.mock
    def test_deletion_is_scoped_to_this_signature(self, sql: SqlClient) -> None:
        # Without the signature filter, re-extracting under a new taxonomy would
        # wipe the previous signature's rows too.
        respx.put(url__startswith=f"{HOST}/api/2.0/fs/files/").mock(httpx.Response(204))
        route = respx.post(STATEMENTS_URL).mock(_ok())

        write_silver(sql, [_outcome("NCT00000001")], batch="b0", signature=SIG)

        delete = next(s for s in _statements(route) if s.strip().startswith("DELETE FROM"))
        assert f"signature = '{SIG}'" in delete

    @respx.mock
    def test_failures_are_recorded_with_an_attempt_counter(self, sql: SqlClient) -> None:
        route = respx.post(STATEMENTS_URL).mock(_ok())

        write_silver(sql, [_failure("NCT00000001")], batch="b0", signature=SIG)

        merge = next(s for s in _statements(route) if "MERGE INTO" in s)
        assert FAILURES_TABLE in merge
        assert "attempts + 1" in merge

    @respx.mock
    def test_a_trial_that_now_succeeds_stops_counting_as_failed(self, sql: SqlClient) -> None:
        respx.put(url__startswith=f"{HOST}/api/2.0/fs/files/").mock(httpx.Response(204))
        route = respx.post(STATEMENTS_URL).mock(_ok())

        write_silver(sql, [_outcome("NCT00000001")], batch="b0", signature=SIG)

        cleanup = [s for s in _statements(route) if FAILURES_TABLE in s and "DELETE" in s]
        assert cleanup and "NCT00000001" in cleanup[0]

    @respx.mock
    def test_a_batch_of_only_failures_writes_no_rows(self, sql: SqlClient) -> None:
        upload = respx.put(url__startswith=f"{HOST}/api/2.0/fs/files/").mock(httpx.Response(204))
        respx.post(STATEMENTS_URL).mock(_ok())

        assert write_silver(sql, [_failure("NCT00000001")], batch="b0", signature=SIG) == (0, 0)
        assert upload.call_count == 0


class TestQuoting:
    """Error text comes back from an HTTP layer and lands in a SQL literal."""

    def test_single_quotes_are_escaped(self) -> None:
        assert _sql_string("it's broken") == "'it''s broken'"

    def test_injection_attempt_is_neutralised(self) -> None:
        quoted = _sql_string("x'; DROP TABLE silver_criteria; --")
        assert quoted.startswith("'") and quoted.endswith("'")
        assert "''" in quoted

    def test_long_errors_are_truncated(self) -> None:
        assert len(_sql_string("e" * 5000)) <= 402

    def test_quote_list_accepts_only_nct_ids(self) -> None:
        assert _quote_list(["NCT00000001", "'; DROP TABLE x; --"]) == "'NCT00000001'"

    def test_quote_list_on_empty_input_is_still_valid_sql(self) -> None:
        assert _quote_list([]) == "''"


class TestStats:
    @respx.mock
    def test_reports_progress_counters(self, sql: SqlClient) -> None:
        respx.post(STATEMENTS_URL).mock(_ok([[3000, 90000, 12, 11200]]))
        assert silver_stats(sql, SIG) == {
            "trials": 3000,
            "criteria": 90000,
            "failed": 12,
            "total": 11200,
        }

    @respx.mock
    def test_empty_result_is_all_zero(self, sql: SqlClient) -> None:
        respx.post(STATEMENTS_URL).mock(_ok([]))
        assert silver_stats(sql, SIG)["trials"] == 0

    @respx.mock
    def test_nulls_are_treated_as_zero(self, sql: SqlClient) -> None:
        respx.post(STATEMENTS_URL).mock(_ok([[None, None, None, 11200]]))
        assert silver_stats(sql, SIG) == {
            "trials": 0,
            "criteria": 0,
            "failed": 0,
            "total": 11200,
        }


class TestSchemaSetup:
    @respx.mock
    def test_ddl_is_idempotent(self, sql: SqlClient) -> None:
        route = respx.post(STATEMENTS_URL).mock(_ok())
        ensure_silver(sql)
        assert all("IF NOT EXISTS" in s for s in _statements(route))
