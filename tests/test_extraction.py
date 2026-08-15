"""Extraction tests.

Two things carry real risk here: the JSON Schema must stay within what the
Foundation Model API accepts, and the response parser must survive both content
shapes the endpoints emit.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from pydantic import BaseModel

from scrutatio.config import Settings
from scrutatio.extraction import (
    Criterion,
    EligibilityExtractor,
    ExtractedEligibility,
    ExtractionError,
    SchemaTooLargeError,
    TruncatedResponseError,
    flatten_schema,
    response_format,
    token_budget,
)
from scrutatio.extraction.client import (
    MAX_TOKEN_CEILING,
    MIN_MAX_TOKENS,
    _message_text,
)

HOST = "https://dbc-00000000-0000.cloud.databricks.com"
TOKEN = "fake-token-for-tests-only"
ENDPOINT_URL = f"{HOST}/serving-endpoints/databricks-gpt-oss-120b/invocations"

RESULT = {
    "criteria": [
        {
            "criterion_id": "inc-1",
            "text": "Stage IV non-small cell lung cancer.",
            "kind": "condition",
            "is_exclusion": False,
        },
        {
            "criterion_id": "exc-1",
            "text": "Untreated CNS metastases.",
            "kind": "comorbidity",
            "is_exclusion": True,
        },
    ]
}


@pytest.fixture
def settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        databricks_host=HOST,
        databricks_token=TOKEN,
        max_retries=2,
    )


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scrutatio.extraction.client.time.sleep", lambda _: None)


def _envelope(content: Any) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class TestSchemaFlattening:
    def test_no_unsupported_keywords_survive(self) -> None:
        # The API rejects $ref, $defs, anyOf, oneOf, allOf, pattern, prefixItems.
        blob = json.dumps(flatten_schema(ExtractedEligibility))
        for forbidden in ("$ref", "$defs", "anyOf", "oneOf", "allOf", "prefixItems"):
            assert forbidden not in blob

    def test_nested_models_are_inlined(self) -> None:
        schema = flatten_schema(ExtractedEligibility)
        item = schema["properties"]["criteria"]["items"]
        assert item["type"] == "object"
        assert set(item["properties"]) == {"criterion_id", "text", "kind", "is_exclusion"}

    def test_objects_are_closed_and_fully_required(self) -> None:
        schema = flatten_schema(ExtractedEligibility)
        assert schema["additionalProperties"] is False
        item = schema["properties"]["criteria"]["items"]
        assert item["additionalProperties"] is False
        assert set(item["required"]) == set(item["properties"])

    def test_kind_is_an_enum_not_a_ref(self) -> None:
        schema = flatten_schema(ExtractedEligibility)
        kind = schema["properties"]["criteria"]["items"]["properties"]["kind"]
        assert "biomarker" in kind["enum"]

    def test_oversized_schema_is_rejected_locally(self) -> None:
        # Better a clear error here than an opaque 400 mid-backfill.
        fields = {f"field_{i}": (str, ...) for i in range(40)}
        big = type("Big", (BaseModel,), {"__annotations__": {k: v[0] for k, v in fields.items()}})
        with pytest.raises(SchemaTooLargeError, match="over the 64"):
            flatten_schema(big)

    def test_response_format_is_strict(self) -> None:
        rf = response_format(ExtractedEligibility, "eligibility")
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["name"] == "eligibility"


class TestMessageShapes:
    def test_plain_string_content(self) -> None:
        assert _message_text({"content": '{"criteria": []}'}) == '{"criteria": []}'

    def test_reasoning_model_array_content(self) -> None:
        # gpt-oss returns [{"type": "reasoning", ...}, {"type": "text", ...}].
        message = {
            "content": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]},
                {"type": "text", "text": '{"criteria": []}'},
            ]
        }
        assert _message_text(message) == '{"criteria": []}'

    def test_reasoning_is_discarded_not_concatenated(self) -> None:
        message = {
            "content": [
                {"type": "reasoning", "text": "SHOULD NOT APPEAR"},
                {"type": "text", "text": "answer"},
            ]
        }
        assert _message_text(message) == "answer"

    def test_unknown_shape_raises(self) -> None:
        with pytest.raises(ExtractionError, match="Could not find assistant text"):
            _message_text({"content": 42})

    def test_array_without_text_part_raises(self) -> None:
        with pytest.raises(ExtractionError):
            _message_text({"content": [{"type": "reasoning", "text": "only thinking"}]})


class TestExtract:
    @respx.mock
    def test_happy_path(self, settings: Settings) -> None:
        respx.post(ENDPOINT_URL).mock(httpx.Response(200, json=_envelope(json.dumps(RESULT))))

        with EligibilityExtractor(settings) as ex:
            result = ex.extract("INCLUSION: stage IV NSCLC. EXCLUSION: untreated CNS mets.")

        assert len(result.criteria) == 2
        assert result.criteria[0].kind == "condition"
        assert result.criteria[1].is_exclusion is True

    @respx.mock
    def test_works_with_reasoning_model_output(self, settings: Settings) -> None:
        content = [
            {"type": "reasoning", "text": "let me think"},
            {"type": "text", "text": json.dumps(RESULT)},
        ]
        respx.post(ENDPOINT_URL).mock(httpx.Response(200, json=_envelope(content)))

        with EligibilityExtractor(settings) as ex:
            assert len(ex.extract("text").criteria) == 2

    def test_empty_input_short_circuits(self, settings: Settings) -> None:
        # No criteria means no API call at all.
        with EligibilityExtractor(settings) as ex:
            assert ex.extract("   ").criteria == []

    @respx.mock
    def test_sends_strict_schema_and_zero_temperature(self, settings: Settings) -> None:
        route = respx.post(ENDPOINT_URL).mock(
            httpx.Response(200, json=_envelope(json.dumps({"criteria": []})))
        )

        with EligibilityExtractor(settings) as ex:
            ex.extract("some text")

        sent = json.loads(route.calls[0].request.content)
        assert sent["temperature"] == 0
        assert sent["response_format"]["json_schema"]["strict"] is True
        assert sent["max_tokens"] > 0

    @respx.mock
    def test_token_is_sent_as_bearer(self, settings: Settings) -> None:
        route = respx.post(ENDPOINT_URL).mock(
            httpx.Response(200, json=_envelope(json.dumps({"criteria": []})))
        )

        with EligibilityExtractor(settings) as ex:
            ex.extract("some text")

        assert route.calls[0].request.headers["authorization"] == f"Bearer {TOKEN}"

    @respx.mock
    def test_schema_invalid_json_is_reported(self, settings: Settings) -> None:
        respx.post(ENDPOINT_URL).mock(
            httpx.Response(200, json=_envelope('{"criteria": [{"bogus": 1}]}'))
        )

        with EligibilityExtractor(settings) as ex, pytest.raises(ExtractionError, match="invalid"):
            ex.extract("text")

    @respx.mock
    def test_unexpected_envelope_is_reported(self, settings: Settings) -> None:
        respx.post(ENDPOINT_URL).mock(httpx.Response(200, json={"nope": True}))

        with EligibilityExtractor(settings) as ex, pytest.raises(ExtractionError, match="envelope"):
            ex.extract("text")


class TestTruncation:
    """Found against the live endpoint: a criteria-heavy trial blew the token
    ceiling and the JSON stopped mid-field. Strict mode was fine — the answer
    was simply cut off, so it must not be reported as a schema violation.
    """

    @respx.mock
    def test_truncation_is_reported_as_truncation(self, settings: Settings) -> None:
        respx.post(ENDPOINT_URL).mock(
            httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": '{"criteria": [{"is_exclusion":'},
                            "finish_reason": "length",
                        }
                    ]
                },
            )
        )

        with (
            EligibilityExtractor(settings, max_tokens=MAX_TOKEN_CEILING) as ex,
            pytest.raises(TruncatedResponseError, match="token ceiling"),
        ):
            ex.extract("text")

    @respx.mock
    def test_retries_once_with_a_doubled_budget(self, settings: Settings) -> None:
        route = respx.post(ENDPOINT_URL)
        route.side_effect = [
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "{"}, "finish_reason": "length"}],
                },
            ),
            httpx.Response(200, json=_envelope(json.dumps(RESULT))),
        ]

        with EligibilityExtractor(settings, max_tokens=1000) as ex:
            assert len(ex.extract("text").criteria) == 2

        first, second = (json.loads(c.request.content) for c in route.calls)
        assert second["max_tokens"] == first["max_tokens"] * 2

    @respx.mock
    def test_does_not_retry_past_the_ceiling(self, settings: Settings) -> None:
        route = respx.post(ENDPOINT_URL).mock(
            httpx.Response(
                200,
                json={"choices": [{"message": {"content": "{"}, "finish_reason": "length"}]},
            )
        )

        with (
            EligibilityExtractor(settings, max_tokens=MAX_TOKEN_CEILING) as ex,
            pytest.raises(TruncatedResponseError),
        ):
            ex.extract("text")

        assert route.call_count == 1

    def test_budget_scales_with_input_length(self) -> None:
        assert token_budget("x" * 100) == MIN_MAX_TOKENS  # floor for short trials
        assert token_budget("x" * 20_000) > MIN_MAX_TOKENS
        assert token_budget("x" * 10_000_000) == MAX_TOKEN_CEILING  # capped

    @respx.mock
    def test_budget_is_derived_from_the_text_by_default(self, settings: Settings) -> None:
        route = respx.post(ENDPOINT_URL).mock(
            httpx.Response(200, json=_envelope(json.dumps({"criteria": []})))
        )
        long_text = "INCLUSION: " + ("a criterion. " * 2000)

        with EligibilityExtractor(settings) as ex:
            ex.extract(long_text)

        assert json.loads(route.calls[0].request.content)["max_tokens"] == token_budget(long_text)


class TestRetries:
    @respx.mock
    def test_retries_on_429(self, settings: Settings, no_sleep: None) -> None:
        route = respx.post(ENDPOINT_URL)
        route.side_effect = [
            httpx.Response(429),
            httpx.Response(200, json=_envelope(json.dumps({"criteria": []}))),
        ]

        with EligibilityExtractor(settings) as ex:
            ex.extract("text")
        assert route.call_count == 2

    @respx.mock
    def test_honours_retry_after_header(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[float] = []
        monkeypatch.setattr("scrutatio.extraction.client.time.sleep", seen.append)

        route = respx.post(ENDPOINT_URL)
        route.side_effect = [
            httpx.Response(429, headers={"retry-after": "7"}),
            httpx.Response(200, json=_envelope(json.dumps({"criteria": []}))),
        ]

        with EligibilityExtractor(settings) as ex:
            ex.extract("text")
        assert seen == [7.0]

    @respx.mock
    def test_malformed_retry_after_falls_back_to_backoff(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[float] = []
        monkeypatch.setattr("scrutatio.extraction.client.time.sleep", seen.append)

        route = respx.post(ENDPOINT_URL)
        route.side_effect = [
            httpx.Response(503, headers={"retry-after": "soon"}),
            httpx.Response(200, json=_envelope(json.dumps({"criteria": []}))),
        ]

        with EligibilityExtractor(settings) as ex:
            ex.extract("text")
        assert seen == [1.0]

    @respx.mock
    def test_400_is_not_retried(self, settings: Settings) -> None:
        route = respx.post(ENDPOINT_URL).mock(httpx.Response(400, text="bad schema"))

        with EligibilityExtractor(settings) as ex, pytest.raises(ExtractionError, match="400"):
            ex.extract("text")
        assert route.call_count == 1

    @respx.mock
    def test_gives_up_eventually(self, settings: Settings, no_sleep: None) -> None:
        route = respx.post(ENDPOINT_URL).mock(httpx.Response(503))

        with (
            EligibilityExtractor(settings) as ex,
            pytest.raises(ExtractionError, match="failed after"),
        ):
            ex.extract("text")
        assert route.call_count == settings.max_retries + 1

    @respx.mock
    def test_network_error_is_retried(self, settings: Settings, no_sleep: None) -> None:
        route = respx.post(ENDPOINT_URL)
        route.side_effect = [
            httpx.ConnectError("boom"),
            httpx.Response(200, json=_envelope(json.dumps({"criteria": []}))),
        ]

        with EligibilityExtractor(settings) as ex:
            ex.extract("text")
        assert route.call_count == 2


class TestConfiguration:
    def test_refuses_to_start_without_credentials(self) -> None:
        bare = Settings(_env_file=None)  # type: ignore[call-arg]
        with pytest.raises(ValueError, match="DATABRICKS_HOST"):
            EligibilityExtractor(bare)

    def test_injected_client_is_not_closed(self, settings: Settings) -> None:
        external = httpx.Client()
        EligibilityExtractor(settings, client=external).close()
        assert not external.is_closed
        external.close()

    def test_criterion_model_round_trips(self) -> None:
        c = Criterion(criterion_id="inc-1", text="x", kind="lab", is_exclusion=False)
        assert Criterion.model_validate(c.model_dump()) == c
