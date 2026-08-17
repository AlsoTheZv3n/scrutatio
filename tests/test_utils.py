"""The shared helpers, and the API's failure translation.

Both halves of ``scrutatio.utils`` are tested here because both are the kind of
code whose bugs are silent. A digest that changes does not raise — it re-queues
351,745 rows. An exception handler that is never exercised looks fine until the
day it has to run.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from scrutatio.utils.hashing import stable_digest, text_digest

pytest.importorskip("fastapi", reason="the `serve` extra is not installed")

from fastapi.testclient import TestClient

from scrutatio.app import create_app, get_db
from scrutatio.clients.openrouter import OpenRouterError
from scrutatio.embeddings import EncoderUnavailableError
from scrutatio.storage.bronze import ensure_storage
from scrutatio.storage.db import IN_MEMORY, connect
from scrutatio.storage.gold import ensure_gold
from scrutatio.storage.silver import ensure_silver
from scrutatio.utils.errors import REQUEST_ID_HEADER, ApiError

# A payload that is byte-identical under `ensure_ascii=False` and different under
# the default. See TestDigestsAreFrozen for why that matters.
_EM_DASH_PAYLOAD = {"prompt": "a criterion — with an em dash", "n": 1}


class TestStableDigest:
    def test_key_order_does_not_change_the_digest(self) -> None:
        # Otherwise reordering a dict literal in the source would invalidate every
        # cached row that depended on it — a diff with no visible behaviour change
        # would quietly re-queue the corpus.
        a = stable_digest({"model": "x", "reasoning": False, "schema": {"b": 1, "a": 2}})
        b = stable_digest({"schema": {"a": 2, "b": 1}, "reasoning": False, "model": "x"})
        assert a == b

    def test_a_changed_value_changes_the_digest(self) -> None:
        assert stable_digest({"reasoning": False}) != stable_digest({"reasoning": True})

    def test_length_is_configurable_and_is_a_prefix(self) -> None:
        full = stable_digest(_EM_DASH_PAYLOAD, length=64)
        assert stable_digest(_EM_DASH_PAYLOAD, length=16) == full[:16]


class TestDigestsAreFrozen:
    """The digest is a storage key, so its value is part of the schema.

    ``stable_digest`` pins ``ensure_ascii=False``. Flipping it changes the bytes
    hashed for any payload containing non-ASCII text — and the extraction
    SYSTEM_PROMPT does. Every one of the 351,745 Silver rows would then read as
    belonging to a different extraction and re-queue, at roughly seven hours and
    three dollars. This test exists so that change cannot be made by accident.
    """

    def test_non_ascii_payload_digest_is_unchanged(self) -> None:
        assert stable_digest(_EM_DASH_PAYLOAD) == "ebf95cb1dca02ebd"

    def test_text_digest_is_unchanged(self) -> None:
        assert text_digest("a b c") == "0e9f64031fcb2bc708b531c2a2044158"


class TestTextDigest:
    def test_whitespace_and_case_are_normalised(self) -> None:
        # The same vignette pasted with a different line wrap is the same
        # question, and paying for a second round of model calls would be a
        # surprise rather than a feature.
        assert text_digest("A  B\nC") == text_digest("a b c")

    def test_different_text_differs(self) -> None:
        assert text_digest("EGFR+ NSCLC") != text_digest("ALK+ NSCLC")


@pytest.fixture
def failing_client() -> Iterator[TestClient]:
    """An app with routes that raise, to exercise the real handler wiring.

    ``raise_server_exceptions=False`` is required for the catch-all: Starlette's
    ServerErrorMiddleware calls the handler and then re-raises, so with the
    default the test would see the exception instead of the 500 the browser gets.

    ``get_db`` is overridden with an in-memory database. Without it these tests
    open ``data/scrutatio.duckdb`` for real, which does not exist in CI and is
    exclusively locked whenever a pipeline run is in flight — the validation test
    below failed with a 503 from the dependency before it ever reached validation.
    """
    conn = connect(IN_MEMORY)
    ensure_storage(conn)
    ensure_silver(conn)
    ensure_gold(conn)

    app = create_app()
    app.dependency_overrides[get_db] = lambda: conn

    @app.get("/_unhandled")
    def _unhandled() -> None:
        raise RuntimeError("postgresql://user:hunter2@host/db leaked in the message")

    @app.get("/_upstream")
    def _upstream() -> None:
        raise OpenRouterError("provider returned 500 after 5 retries")

    @app.get("/_encoder")
    def _encoder() -> None:
        raise EncoderUnavailableError("sentence-transformers is not installed")

    @app.get("/_named")
    def _named() -> None:
        raise ApiError(status_code=503, code="gold_empty", detail="no vectors yet")

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        conn.close()


class TestRequestIds:
    def test_successful_responses_carry_one(self, failing_client: TestClient) -> None:
        response = failing_client.get("/health")
        assert response.status_code == 200
        assert response.headers[REQUEST_ID_HEADER]

    def test_the_body_and_the_header_agree(self, failing_client: TestClient) -> None:
        # The whole point is correlation: a client quoting the id from the body
        # must let us find the header-logged line, and vice versa.
        response = failing_client.get("/_upstream")
        assert response.json()["request_id"] == response.headers[REQUEST_ID_HEADER]

    def test_ids_differ_between_requests(self, failing_client: TestClient) -> None:
        first = failing_client.get("/_upstream").json()["request_id"]
        second = failing_client.get("/_upstream").json()["request_id"]
        assert first != second


class TestTranslation:
    def test_upstream_failure_is_502_not_500(self, failing_client: TestClient) -> None:
        # The distinction is actionable: 502 says the fault is at OpenRouter and
        # retrying may work, 500 says it is ours and retrying will not.
        response = failing_client.get("/_upstream")
        assert response.status_code == 502
        assert response.json()["code"] == "upstream_failed"

    def test_a_missing_encoder_is_503(self, failing_client: TestClient) -> None:
        response = failing_client.get("/_encoder")
        assert response.status_code == 503
        body = response.json()
        assert body["code"] == "encoder_unavailable"
        # The install hint has to survive translation or the message is useless.
        assert "sentence-transformers" in body["detail"]

    def test_a_named_api_error_keeps_its_code(self, failing_client: TestClient) -> None:
        response = failing_client.get("/_named")
        assert response.status_code == 503
        assert response.json()["code"] == "gold_empty"


class TestUnhandledFailures:
    def test_it_becomes_a_500_with_a_code(self, failing_client: TestClient) -> None:
        response = failing_client.get("/_unhandled")
        assert response.status_code == 500
        assert response.json()["code"] == "internal_error"

    def test_the_exception_message_is_not_leaked(self, failing_client: TestClient) -> None:
        # An unmapped failure may carry a connection string, a prompt fragment or
        # a key in its message. The request id is how the caller and the log are
        # joined instead.
        body = failing_client.get("/_unhandled").text
        assert "hunter2" not in body
        assert "postgresql://" not in body

    def test_it_still_tells_the_caller_what_to_quote(self, failing_client: TestClient) -> None:
        body = failing_client.get("/_unhandled").json()
        assert body["request_id"] in body["detail"]


class TestFrameworkErrorsUseTheSameShape:
    def test_a_405_still_says_which_methods_work(self, failing_client: TestClient) -> None:
        # RFC 9110 §15.5.6: a 405 MUST carry Allow. Reshaping the framework's
        # response into ErrorBody dropped it, so the answer to "GET /match" in a
        # browser was "Method Not Allowed" and nothing else.
        response = failing_client.get("/match")
        assert response.status_code == 405
        assert response.headers.get("allow"), "the mandatory Allow header was dropped"
        assert "POST" in response.headers["allow"]

    def test_a_405_names_the_method_in_the_detail_too(self, failing_client: TestClient) -> None:
        # The header is for machines; a person typing the URL sees only the body.
        body = failing_client.get("/match").json()
        assert body["code"] == "http_405"
        assert "POST" in body["detail"]
        assert "/match" in body["detail"]

    def test_an_unknown_route_is_reshaped(self, failing_client: TestClient) -> None:
        # One error format, so the frontend needs one branch rather than two.
        response = failing_client.get("/does-not-exist")
        assert response.status_code == 404
        body = response.json()
        assert body["code"] == "http_404"
        assert body["request_id"]

    def test_validation_failures_carry_the_field_path(self, failing_client: TestClient) -> None:
        response = failing_client.post("/match", json={"text": "hi"})
        assert response.status_code == 422
        body = response.json()
        assert body["code"] == "validation_error"
        # Flattened rather than structured, but the field must still be nameable
        # or the message cannot be acted on.
        assert "text" in body["detail"]
