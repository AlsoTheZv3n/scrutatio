"""Client tests.

The interesting cases are the three API behaviours that fail silently rather
than loudly: dropped query parameters on page 2, the nested last-update date,
and the RootModel wrapper around the study list.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx
import pytest
import respx

from scrutatio.config import Settings
from scrutatio.ctgov import CtGovClient, CtGovError, last_update_posted, nct_id
from scrutatio.ctgov.models import Study

STUDIES_URL = "https://clinicaltrials.gov/api/v2/studies"


def _study(nct_id: str, updated: str | None = "2026-08-14") -> dict[str, Any]:
    status: dict[str, Any] = {"overallStatus": "RECRUITING"}
    if updated is not None:
        status["lastUpdatePostDateStruct"] = {"date": updated, "type": "ACTUAL"}
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id},
            "statusModule": status,
            "eligibilityModule": {"eligibilityCriteria": "INCLUSION: adult."},
        }
    }


def _page(studies: list[dict[str, Any]], token: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"studies": studies}
    if token is not None:
        payload["nextPageToken"] = token
    return payload


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, page_size=2, max_retries=2)  # type: ignore[call-arg]


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scrutatio.ctgov.client.time.sleep", lambda _: None)


class TestPagination:
    @respx.mock
    def test_follows_next_page_token(self, settings: Settings) -> None:
        route = respx.get(STUDIES_URL)
        route.side_effect = [
            httpx.Response(
                200, json=_page([_study("NCT00000001"), _study("NCT00000002")], token="TOK")
            ),
            httpx.Response(200, json=_page([_study("NCT00000003")])),
        ]

        with CtGovClient(settings) as client:
            studies = list(client.iter_studies())

        assert len(studies) == 3
        assert route.call_count == 2

    @respx.mock
    def test_repeats_the_query_on_page_two(self, settings: Settings) -> None:
        # Dropping filter params on page 2 silently re-queries the whole registry.
        route = respx.get(STUDIES_URL)
        route.side_effect = [
            httpx.Response(200, json=_page([_study("NCT00000001")], token="TOK")),
            httpx.Response(200, json=_page([_study("NCT00000002")])),
        ]

        with CtGovClient(settings) as client:
            list(client.iter_studies())

        first, second = (call.request.url.params for call in route.calls)
        assert second["filter.advanced"] == first["filter.advanced"]
        assert second["filter.overallStatus"] == "RECRUITING"
        assert second["pageToken"] == "TOK"
        assert "pageToken" not in first

    @respx.mock
    def test_caller_cannot_smuggle_paging_params(self, settings: Settings) -> None:
        route = respx.get(STUDIES_URL).mock(
            httpx.Response(200, json=_page([_study("NCT00000001")]))
        )

        with CtGovClient(settings) as client:
            list(client.iter_studies(query={"query.cond": "cancer", "pageToken": "NOPE"}))

        params = route.calls[0].request.url.params
        assert params["query.cond"] == "cancer"
        assert "NOPE" not in str(params)

    @respx.mock
    def test_limit_stops_early(self, settings: Settings) -> None:
        respx.get(STUDIES_URL).mock(
            httpx.Response(
                200, json=_page([_study("NCT00000001"), _study("NCT00000002")], token="TOK")
            )
        )

        with CtGovClient(settings) as client:
            assert len(list(client.iter_studies(limit=1))) == 1

    @respx.mock
    def test_page_size_is_capped_at_1000(self) -> None:
        route = respx.get(STUDIES_URL).mock(httpx.Response(200, json=_page([])))
        settings = Settings(_env_file=None, page_size=1000)  # type: ignore[call-arg]

        with CtGovClient(settings) as client:
            list(client.iter_studies())

        assert route.calls[0].request.url.params["pageSize"] == "1000"


class TestIncremental:
    @respx.mock
    def test_updated_since_appends_a_range_clause(self, settings: Settings) -> None:
        route = respx.get(STUDIES_URL).mock(httpx.Response(200, json=_page([])))

        with CtGovClient(settings) as client:
            list(client.iter_studies(updated_since=date(2026, 8, 1)))

        advanced = route.calls[0].request.url.params["filter.advanced"]
        assert "AREA[LastUpdatePostDate]RANGE[2026-08-01,MAX]" in advanced
        assert "Neoplasms" in advanced  # scope preserved, not replaced

    def test_watermark_reads_the_nested_path(self) -> None:
        # The flat `lastUpdatePostDate` path yields None for every study and
        # freezes the watermark without raising.
        study = Study.model_validate(_study("NCT00000001", updated="2026-08-14"))
        assert last_update_posted(study) == date(2026, 8, 14)

    def test_watermark_is_none_when_absent(self) -> None:
        assert last_update_posted(Study.model_validate(_study("NCT00000001", updated=None))) is None

    def test_watermark_is_none_without_protocol_section(self) -> None:
        assert last_update_posted(Study.model_validate({})) is None


class TestErrorHandling:
    @respx.mock
    def test_400_raises_immediately_without_retry(self, settings: Settings) -> None:
        # An unknown query parameter is a hard 400; retrying cannot help.
        route = respx.get(STUDIES_URL).mock(httpx.Response(400, text="unknown parameter"))

        with CtGovClient(settings) as client, pytest.raises(CtGovError, match="400"):
            list(client.iter_studies())

        assert route.call_count == 1

    @respx.mock
    def test_retries_on_429_then_succeeds(self, settings: Settings, no_sleep: None) -> None:
        route = respx.get(STUDIES_URL)
        route.side_effect = [
            httpx.Response(429),
            httpx.Response(200, json=_page([_study("NCT00000001")])),
        ]

        with CtGovClient(settings) as client:
            assert len(list(client.iter_studies())) == 1
        assert route.call_count == 2

    @respx.mock
    def test_gives_up_after_max_retries(self, settings: Settings, no_sleep: None) -> None:
        route = respx.get(STUDIES_URL).mock(httpx.Response(503))

        with CtGovClient(settings) as client, pytest.raises(CtGovError, match="failed after"):
            list(client.iter_studies())

        assert route.call_count == settings.max_retries + 1

    @respx.mock
    def test_retries_on_network_error(self, settings: Settings, no_sleep: None) -> None:
        route = respx.get(STUDIES_URL)
        route.side_effect = [
            httpx.ConnectError("boom"),
            httpx.Response(200, json=_page([_study("NCT00000001")])),
        ]

        with CtGovClient(settings) as client:
            assert len(list(client.iter_studies())) == 1

    @respx.mock
    def test_schema_mismatch_is_reported(self, settings: Settings) -> None:
        respx.get(STUDIES_URL).mock(httpx.Response(200, json={"unexpected": True}))

        with CtGovClient(settings) as client, pytest.raises(CtGovError, match="schema"):
            list(client.iter_studies())


class TestSingleStudyAndCount:
    @respx.mock
    def test_fetch_study(self, settings: Settings) -> None:
        respx.get(f"{STUDIES_URL}/NCT00000001").mock(
            httpx.Response(200, json=_study("NCT00000001"))
        )

        with CtGovClient(settings) as client:
            study = client.fetch_study("NCT00000001")

        assert nct_id(study) == "NCT00000001"

    def test_nct_id_unwraps_the_root_model(self) -> None:
        # The raw attribute is an Nct RootModel; comparing it to a str fails silently.
        assert nct_id(Study.model_validate(_study("NCT00000001"))) == "NCT00000001"

    def test_nct_id_is_none_without_identification_module(self) -> None:
        assert nct_id(Study.model_validate({})) is None

    @respx.mock
    def test_fetch_study_rejects_bad_payload(self, settings: Settings) -> None:
        respx.get(f"{STUDIES_URL}/NCT00000001").mock(
            httpx.Response(200, json={"protocolSection": 5})
        )

        with CtGovClient(settings) as client, pytest.raises(CtGovError, match="schema"):
            client.fetch_study("NCT00000001")

    @respx.mock
    def test_count_requests_total_only(self, settings: Settings) -> None:
        route = respx.get(STUDIES_URL).mock(httpx.Response(200, json={"totalCount": 11200}))

        with CtGovClient(settings) as client:
            assert client.count() == 11200

        params = route.calls[0].request.url.params
        assert params["countTotal"] == "true"
        assert params["pageSize"] == "1"


class TestScopeQuery:
    @respx.mock
    def test_scope_includes_recruiting_filter(self, settings: Settings) -> None:
        route = respx.get(STUDIES_URL).mock(httpx.Response(200, json=_page([])))

        with CtGovClient(settings) as client:
            list(client.iter_studies())

        params = route.calls[0].request.url.params
        assert params["filter.overallStatus"] == "RECRUITING"
        assert "AREA[ConditionAncestorTerm]Neoplasms" in params["filter.advanced"]

    @respx.mock
    def test_recruiting_filter_can_be_disabled(self) -> None:
        settings = Settings(_env_file=None, recruiting_only=False)  # type: ignore[call-arg]
        route = respx.get(STUDIES_URL).mock(httpx.Response(200, json=_page([])))

        with CtGovClient(settings) as client:
            list(client.iter_studies())

        assert "filter.overallStatus" not in route.calls[0].request.url.params

    def test_injected_client_is_not_closed(self, settings: Settings) -> None:
        external = httpx.Client()
        CtGovClient(settings, client=external).close()
        assert not external.is_closed
        external.close()
