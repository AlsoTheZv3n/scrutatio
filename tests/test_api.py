"""API tests.

The interesting cases are the honest-failure ones. This service sits on a
single-writer database and in front of layers that may not be built yet, so
"cannot answer, and here is why" is a first-class response rather than an edge
case — and it is what the frontend has to render.
"""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
import pytest

pytest.importorskip("fastapi", reason="the `serve` extra is not installed")

from fastapi.testclient import TestClient

from scrutatio.app import create_app, get_db
from scrutatio.storage.bronze import BRONZE_TABLE, ensure_storage
from scrutatio.storage.db import IN_MEMORY, DatabaseBusyError, connect
from scrutatio.storage.gold import ensure_gold
from scrutatio.storage.silver import (
    SILVER_TABLE,
    ensure_silver,
    extraction_signature,
)

VITE_ORIGIN = "http://localhost:5173"


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = connect(IN_MEMORY)
    ensure_storage(conn)
    ensure_silver(conn)
    ensure_gold(conn)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def client(db: duckdb.DuckDBPyConnection) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _land(db: duckdb.DuckDBPyConnection, nct: str, ordinal: int, kind: str, excl: bool) -> None:
    db.execute(
        f"INSERT INTO {SILVER_TABLE} VALUES (?, ?, ?, 'c', ?, ?, ?, current_timestamp)",  # noqa: S608
        [nct, extraction_signature(), ordinal, f"criterion {ordinal}", kind, excl],
    )


class TestIndex:
    """The base URL must answer. A bare 404 there is the first thing anyone sees."""

    def test_the_root_lists_the_endpoints(self) -> None:
        with TestClient(create_app()) as c:
            r = c.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "Scrutatio"
        # Every route the app actually serves must be named here, or the index is
        # documentation that drifts.
        assert "POST /match" in body["endpoints"]
        assert "GET /corpus/stats" in body["endpoints"]

    def test_it_carries_the_disclaimer(self) -> None:
        # Required by the Definition of Done, and cheaper to guarantee here than to
        # rely on every consumer remembering.
        with TestClient(create_app()) as c:
            body = c.get("/").json()
        assert "not medical advice" in body["disclaimer"]

    def test_it_answers_without_touching_the_database(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def busy(*_: object, **__: object) -> None:
            raise DatabaseBusyError("held")

        monkeypatch.setattr("scrutatio.app.database", busy)
        with TestClient(create_app()) as c:
            assert c.get("/").status_code == 200


class TestHealth:
    def test_it_answers_without_touching_the_database(self) -> None:
        # Deliberate: a pipeline run holds the file exclusively, and liveness
        # must not depend on winning that race.
        with TestClient(create_app()) as c:
            body = c.get("/health").json()
        assert body["status"] == "ok"
        assert len(body["signature"]) == 16


class TestBusyDatabase:
    """DuckDB allows one process. A request arriving mid-run must be told so,
    not left hanging and not shown a traceback."""

    def test_data_endpoints_report_503_with_the_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def busy(*_: object, **__: object) -> None:
            raise DatabaseBusyError("data/scrutatio.duckdb is held by another process")

        monkeypatch.setattr("scrutatio.app.database", busy)
        with TestClient(create_app()) as c:
            r = c.get("/corpus/stats")

        assert r.status_code == 503
        # Translated by the handler table in scrutatio.utils.errors rather than by
        # a try/except in the dependency, so the code is machine-readable and the
        # wording comes from the exception that actually knows what happened.
        body = r.json()
        assert body["code"] == "database_busy"
        assert "another process" in body["detail"]
        assert body["request_id"]


class TestCors:
    def test_the_vite_dev_origin_is_allowed(self, client: TestClient) -> None:
        r = client.options(
            "/corpus/stats",
            headers={"Origin": VITE_ORIGIN, "Access-Control-Request-Method": "GET"},
        )
        assert r.headers["access-control-allow-origin"] == VITE_ORIGIN

    def test_a_foreign_origin_is_not(self, client: TestClient) -> None:
        # Listed explicitly rather than "*": no third-party page should be able
        # to drive this service.
        r = client.options(
            "/corpus/stats",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
        )
        assert "access-control-allow-origin" not in r.headers


class TestCorpusStats:
    def test_reports_counts_per_layer(
        self, client: TestClient, db: duckdb.DuckDBPyConnection
    ) -> None:
        db.execute(
            f"INSERT INTO {BRONZE_TABLE} VALUES ('NCT00000001', DATE '2026-08-14', '{{}}', current_timestamp)"  # noqa: S608, E501
        )
        _land(db, "NCT00000001", 0, "condition", False)

        body = client.get("/corpus/stats").json()
        assert body["bronze_trials"] == 1
        assert body["newest_update"] == "2026-08-14"
        assert body["silver_criteria"] == 1
        assert body["gold_vectors"] == 0
        assert body["embedding_model"].startswith("BAAI/bge-")


class TestCorpusCriteria:
    def test_returns_a_page_with_a_total(
        self, client: TestClient, db: duckdb.DuckDBPyConnection
    ) -> None:
        for i in range(5):
            _land(db, "NCT00000001", i, "condition", False)

        body = client.get("/corpus/criteria?limit=2").json()
        assert body["total"] == 5
        assert len(body["rows"]) == 2
        assert body["limit"] == 2

    def test_filters_by_kind(self, client: TestClient, db: duckdb.DuckDBPyConnection) -> None:
        _land(db, "NCT00000001", 0, "condition", False)
        _land(db, "NCT00000001", 1, "consent", False)

        body = client.get("/corpus/criteria?kind=consent").json()
        assert body["total"] == 1
        assert body["rows"][0]["kind"] == "consent"

    def test_filters_by_exclusion(self, client: TestClient, db: duckdb.DuckDBPyConnection) -> None:
        _land(db, "NCT00000001", 0, "condition", False)
        _land(db, "NCT00000001", 1, "condition", True)

        body = client.get("/corpus/criteria?is_exclusion=true").json()
        assert body["total"] == 1
        assert body["rows"][0]["is_exclusion"] is True

    def test_the_page_size_is_bounded(self, client: TestClient) -> None:
        assert client.get("/corpus/criteria?limit=100000").status_code == 422


class TestMatch:
    def test_says_what_is_missing_rather_than_returning_a_stub(
        self, client: TestClient, db: duckdb.DuckDBPyConnection
    ) -> None:
        # A stub that looks like a result is worse than a refusal: the frontend
        # would render it as an answer.
        _land(db, "NCT00000001", 0, "condition", False)

        r = client.post("/match", json={"text": "58-year-old woman with EGFR+ NSCLC."})
        assert r.status_code == 503
        detail = r.json()["detail"]
        assert "Gold layer holds no vectors" in detail
        assert "scrutatio embed" in detail

    def test_too_short_a_description_is_rejected(self, client: TestClient) -> None:
        assert client.post("/match", json={"text": "hi"}).status_code == 422
