"""Live ClinicalTrials.gov contract.

Free, unauthenticated, and guarding the most brittle assumption in the codebase:
``src/scrutatio/clients/ctgov/models.py`` is generated from an OpenAPI schema and
every model in it sets ``extra="forbid"``. One new field in the registry's
response and parsing raises — on every study, for the whole backfill.

Nothing else can find that. The mocked tests feed fixtures we wrote, so they
agree with the models by construction. Only a real response disagrees.
"""

from __future__ import annotations

import pytest

from scrutatio.clients.ctgov import CtGovClient, last_update_posted, nct_id

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def studies() -> list:
    """One real page. Each item is already validated — ``iter_studies`` returns
    ``Study`` objects, so a schema drift raises here rather than in an assertion."""
    with CtGovClient() as client:
        return list(client.iter_studies(limit=25))


class TestSchemaDrift:
    def test_a_real_page_still_validates_against_the_generated_models(self, studies: list) -> None:
        # The actual assertion is that the fixture did not raise. 25 studies rather
        # than one, because optional fields only appear in some of them and
        # `extra="forbid"` fires on whichever study carries the new key.
        assert len(studies) == 25

    def test_the_accessors_still_find_their_fields(self, studies: list) -> None:
        # Both walk a nested path the generator produced. A renamed module in the
        # registry's payload would leave these returning None for every study while
        # validation still passed — silent, and it would empty the watermark.
        identifiers = [nct_id(s) for s in studies]
        assert all(i and i.startswith("NCT") for i in identifiers), identifiers[:3]
        assert sum(last_update_posted(s) is not None for s in studies) >= 20


class TestScopeQuery:
    """``AREA[ConditionAncestorTerm]Neoplasms AND AREA[StudyType]INTERVENTIONAL``
    plus ``filter.overallStatus=RECRUITING``. If that syntax stops being accepted,
    the corpus silently becomes something else.
    """

    def test_the_scope_returns_a_corpus_of_the_expected_order(self) -> None:
        with CtGovClient() as client:
            total = client.count()
        # Measured 11,200 on 2026-08-16. The bound is deliberately wide: the
        # registry grows and trials stop recruiting, so this catches a collapse or a
        # rejected filter, not normal drift.
        assert 5_000 < total < 40_000, f"scope returned {total} studies"

    def test_a_narrower_query_returns_fewer(self) -> None:
        # Guards against the filter being ignored rather than applied — an ignored
        # `filter.advanced` would return the whole registry for both queries.
        with CtGovClient() as client:
            scoped = client.count()
            narrower = client.count(
                query={
                    "filter.advanced": (
                        "AREA[ConditionAncestorTerm]Neoplasms "
                        "AND AREA[StudyType]INTERVENTIONAL "
                        "AND AREA[Phase]PHASE3"
                    ),
                    "filter.overallStatus": "RECRUITING",
                }
            )
        assert 0 < narrower < scoped, f"phase-3 subset {narrower} vs scope {scoped}"


class TestSingleStudy:
    def test_fetching_one_study_by_id_round_trips(self, studies: list) -> None:
        # Uses an id from the live page rather than a hardcoded one: a hardcoded
        # NCT can be withdrawn, and the test would then fail for a reason that has
        # nothing to do with our code.
        identifier = nct_id(studies[0])
        assert identifier is not None

        with CtGovClient() as client:
            fetched = client.fetch_study(identifier)

        assert nct_id(fetched) == identifier
