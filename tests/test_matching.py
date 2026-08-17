"""Matching tests: the cache key, the judge's contract, and the orchestration.

Three properties carry this layer:

* **The cache key covers everything that changes a verdict.** A stale answer that
  looks current is worse than no cache, because nothing in the result says so.
* **Every criterion comes back with a verdict.** A row missing from the response
  reads as "not applicable" in the UI when it means "not answered".
* **One failing candidate must not lose the other nine.** The same defect that
  killed a twelve-hour extraction pass, in a place where it would be visible to a
  user instead of a log.

No network and no model: the encoder and judge are fakes, so what is tested is
our handling rather than a provider's behaviour.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence

import duckdb
import pytest

from scrutatio.clients.openrouter import OpenRouterError
from scrutatio.matching.judge import CriterionJudge, judge_prompt_version
from scrutatio.matching.schema import CriterionVerdict
from scrutatio.pipeline.match import run_match
from scrutatio.storage.bronze import BRONZE_TABLE, ensure_storage, trial_metadata
from scrutatio.storage.cache import (
    cache_stats,
    cached_verdicts,
    ensure_cache,
    query_hash,
    store_verdicts,
)
from scrutatio.storage.db import IN_MEMORY, connect
from scrutatio.storage.gold import ensure_gold, write_embeddings
from scrutatio.storage.silver import (
    SILVER_TABLE,
    ensure_silver,
    extraction_signature,
    trial_criteria,
)

SIG = "abc123def4567890"
MODEL = "BAAI/bge-large-en-v1.5"
VIGNETTE = "58-year-old woman with EGFR exon 19 deletion NSCLC, progression on osimertinib."


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = connect(IN_MEMORY)
    ensure_storage(conn)
    ensure_silver(conn)
    ensure_gold(conn)
    ensure_cache(conn)
    try:
        yield conn
    finally:
        conn.close()


def _verdict(ordinal: int = 0, verdict: str = "met") -> CriterionVerdict:
    return CriterionVerdict(
        ordinal=ordinal,
        text=f"criterion {ordinal}",
        kind="condition",
        is_exclusion=False,
        verdict=verdict,  # type: ignore[arg-type]
        rationale="because the description says so",
    )


class TestQueryHash:
    def test_the_same_vignette_reflowed_is_the_same_question(self) -> None:
        # Otherwise a paste with different line wrapping pays for the same answer
        # a second time, which would be a surprise rather than a feature.
        assert query_hash("EGFR+ NSCLC,\n  progression") == query_hash("egfr+ nsclc, progression")

    def test_a_different_patient_is_a_different_key(self) -> None:
        assert query_hash("stage IV") != query_hash("stage II")


class TestCacheKey:
    """Four parts, each for a reason. A miss on any of them is the point."""

    def _store(self, db: duckdb.DuckDBPyConnection, **kw: str) -> None:
        store_verdicts(
            db,
            kw.get("query_key", query_hash(VIGNETTE)),
            kw.get("nct_id", "NCT00000001"),
            [_verdict()],
            signature=kw.get("signature", SIG),
            prompt_version=kw.get("prompt_version", "v1"),
        )

    def _get(self, db: duckdb.DuckDBPyConnection, **kw: str) -> dict:
        return cached_verdicts(
            db,
            kw.get("query_key", query_hash(VIGNETTE)),
            [kw.get("nct_id", "NCT00000001")],
            signature=kw.get("signature", SIG),
            prompt_version=kw.get("prompt_version", "v1"),
        )

    def test_a_stored_verdict_is_found_again(self, db: duckdb.DuckDBPyConnection) -> None:
        self._store(db)
        found = self._get(db)
        assert found["NCT00000001"][0].verdict == "met"
        assert found["NCT00000001"][0].rationale  # the reason survives the round trip

    def test_a_different_query_misses(self, db: duckdb.DuckDBPyConnection) -> None:
        self._store(db)
        assert self._get(db, query_key=query_hash("someone else")) == {}

    def test_a_different_extraction_signature_misses(self, db: duckdb.DuckDBPyConnection) -> None:
        # The criteria changed, so verdicts about the old criteria are about text
        # that no longer exists.
        self._store(db)
        assert self._get(db, signature="0000111122223333") == {}

    def test_a_different_prompt_version_misses(self, db: duckdb.DuckDBPyConnection) -> None:
        # Editing the judge prompt must not leave answers from the previous one
        # sitting in the cache looking current.
        self._store(db)
        assert self._get(db, prompt_version="v2") == {}

    def test_restoring_the_same_key_updates_rather_than_duplicates(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        self._store(db)
        self._store(db)
        assert cache_stats(db) == {"entries": 1, "queries": 1}

    def test_the_prompt_version_is_stable_and_short(self) -> None:
        assert judge_prompt_version() == judge_prompt_version()
        assert len(judge_prompt_version()) == 16


class TestJudgeContract:
    """The judge's own HTTP path is exercised through a stubbed transport."""

    class _FakeClient:
        def __init__(self, body: object) -> None:
            self._body = body
            self.payloads: list[dict] = []

        def build_payload(self, **kw: object) -> dict:
            return dict(kw)

        def post(self, payload: dict) -> object:
            self.payloads.append(payload)
            return self._body

        def close(self) -> None: ...

    @staticmethod
    def _envelope(verdicts: list[dict]) -> dict:
        return {"choices": [{"message": {"content": json.dumps({"verdicts": verdicts})}}]}

    def test_no_criteria_makes_no_call(self) -> None:
        client = self._FakeClient(self._envelope([]))
        judge = CriterionJudge(client=client)  # type: ignore[arg-type]
        assert judge.judge(VIGNETTE, []) == []
        assert client.payloads == []

    def test_every_criterion_comes_back(self) -> None:
        criteria = [(0, "adult", "age", False), (1, "no brain mets", "condition", True)]
        client = self._FakeClient(
            self._envelope(
                [
                    {"ordinal": 0, "verdict": "met", "rationale": "she is 58"},
                    {"ordinal": 1, "verdict": "not_met", "rationale": "brain mets present"},
                ]
            )
        )
        out = CriterionJudge(client=client).judge(VIGNETTE, criteria)  # type: ignore[arg-type]

        assert [v.ordinal for v in out] == [0, 1]
        assert out[1].verdict == "not_met"
        assert out[1].is_exclusion is True

    def test_an_omitted_criterion_becomes_unclear_and_says_so(self) -> None:
        # Dropping it would read as "not applicable" in the UI. It means the
        # model did not answer, and the difference matters.
        criteria = [(0, "adult", "age", False), (1, "ECOG 0-1", "ecog", False)]
        client = self._FakeClient(
            self._envelope([{"ordinal": 0, "verdict": "met", "rationale": "she is 58"}])
        )
        out = CriterionJudge(client=client).judge(VIGNETTE, criteria)  # type: ignore[arg-type]

        assert len(out) == 2
        assert out[1].verdict == "unclear"
        assert "no verdict" in out[1].rationale.lower()

    def test_the_exclusion_flag_reaches_the_prompt(self) -> None:
        client = self._FakeClient(
            self._envelope([{"ordinal": 0, "verdict": "met", "rationale": "ok"}])
        )
        CriterionJudge(client=client).judge(  # type: ignore[arg-type]
            VIGNETTE, [(0, "no prior chemo", "prior_therapy", True)]
        )
        assert "[EXCLUSION]" in client.payloads[0]["user"]

    def test_schema_invalid_output_is_reported_as_such(self) -> None:
        client = self._FakeClient({"choices": [{"message": {"content": "[not an object]"}}]})
        with pytest.raises(OpenRouterError, match="schema-invalid"):
            CriterionJudge(client=client).judge(  # type: ignore[arg-type]
                VIGNETTE, [(0, "adult", "age", False)]
            )

    def test_an_unexpected_envelope_is_reported(self) -> None:
        client = self._FakeClient({"nope": True})
        with pytest.raises(OpenRouterError, match="envelope"):
            CriterionJudge(client=client).judge(  # type: ignore[arg-type]
                VIGNETTE, [(0, "adult", "age", False)]
            )


class TestReaders:
    def test_trial_metadata_reads_the_stored_payload(self, db: duckdb.DuckDBPyConnection) -> None:
        raw = json.dumps(
            {
                "protocolSection": {
                    "identificationModule": {"briefTitle": "A study of things"},
                    "designModule": {"phases": ["PHASE2", "PHASE3"]},
                    "statusModule": {"overallStatus": "RECRUITING"},
                    "contactsLocationsModule": {
                        "locations": [
                            {"facility": "Clinic", "city": "Bern", "country": "Switzerland"}
                        ]
                    },
                }
            }
        )
        db.execute(
            f"INSERT INTO {BRONZE_TABLE} VALUES ('NCT00000001', NULL, ?, current_timestamp)",  # noqa: S608
            [raw],
        )

        meta = trial_metadata(db, ["NCT00000001"])["NCT00000001"]
        assert meta.title == "A study of things"
        assert meta.phase == "PHASE2, PHASE3"
        assert meta.locations == ["Clinic, Bern, Switzerland"]

    def test_long_location_lists_are_summarised_not_dropped(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        # One real trial carries 1,259 locations. Sending them all would make the
        # payload the largest thing in the response.
        raw = json.dumps(
            {
                "protocolSection": {
                    "identificationModule": {"briefTitle": "Big"},
                    "statusModule": {"overallStatus": "RECRUITING"},
                    "contactsLocationsModule": {
                        "locations": [
                            {"facility": f"Site {i}", "city": "X", "country": "Y"}
                            for i in range(25)
                        ]
                    },
                }
            }
        )
        db.execute(
            f"INSERT INTO {BRONZE_TABLE} VALUES ('NCT00000002', NULL, ?, current_timestamp)",  # noqa: S608
            [raw],
        )

        locations = trial_metadata(db, ["NCT00000002"])["NCT00000002"].locations
        assert len(locations) == 11
        assert locations[-1] == "+15 more"

    def test_trial_criteria_groups_by_trial_in_ordinal_order(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        for nct, ordinal in (("NCT00000001", 1), ("NCT00000001", 0), ("NCT00000002", 0)):
            db.execute(
                f"INSERT INTO {SILVER_TABLE} VALUES (?, ?, ?, 'c', ?, 'condition', false, current_timestamp)",  # noqa: S608, E501
                [nct, SIG, ordinal, f"criterion {ordinal}"],
            )

        grouped = trial_criteria(db, ["NCT00000001", "NCT00000002"], SIG)
        assert [c[0] for c in grouped["NCT00000001"]] == [0, 1]
        assert len(grouped["NCT00000002"]) == 1

    def test_empty_input_reads_nothing(self, db: duckdb.DuckDBPyConnection) -> None:
        assert trial_metadata(db, []) == {}
        assert trial_criteria(db, [], SIG) == {}


class _FakeEncoder:
    model_name = MODEL

    def encode_query(self, text: str) -> list[float]:
        vec = [0.0] * 1024
        vec[0] = 1.0
        return vec


class _RecordingJudge:
    """Echoes every criterion back as `met`, and counts how often it was asked.

    It does not need to know which trial it is judging: the pipeline hands it one
    trial's criteria at a time, and what the tests check is how many calls
    happened and what came back.
    """

    model_name = "fake/model"

    def __init__(self) -> None:
        self.calls = 0

    def judge(
        self, text: str, criteria: Sequence[tuple[int, str, str, bool]]
    ) -> list[CriterionVerdict]:
        self.calls += 1
        return [
            CriterionVerdict(
                ordinal=o,
                text=t,
                kind=k,  # type: ignore[arg-type]
                is_exclusion=x,
                verdict="met",
                rationale="fine",
            )
            for o, t, k, x in criteria
        ]


class TestPipeline:
    def _corpus(self, db: duckdb.DuckDBPyConnection, trials: int = 2) -> None:
        for i in range(trials):
            nct = f"NCT0000000{i}"
            raw = json.dumps(
                {
                    "protocolSection": {
                        "identificationModule": {"briefTitle": f"Trial {i}"},
                        "statusModule": {"overallStatus": "RECRUITING"},
                    }
                }
            )
            db.execute(
                f"INSERT INTO {BRONZE_TABLE} VALUES (?, NULL, ?, current_timestamp)",  # noqa: S608
                [nct, raw],
            )
            for ordinal in range(2):
                db.execute(
                    f"INSERT INTO {SILVER_TABLE} VALUES (?, ?, ?, 'c', ?, 'condition', false, current_timestamp)",  # noqa: S608, E501
                    [nct, extraction_signature(), ordinal, f"criterion {ordinal} of {nct}"],
                )
                vec = [0.0] * 1024
                vec[i] = 1.0
                write_embeddings(
                    db,
                    [(nct, ordinal, vec)],
                    signature=extraction_signature(),
                    model=MODEL,
                )

    def test_it_assembles_a_full_response(self, db: duckdb.DuckDBPyConnection) -> None:
        self._corpus(db)
        judge = _RecordingJudge()

        response, timing = run_match(db=db, encoder=_FakeEncoder(), judge=judge, text=VIGNETTE, k=2)

        assert len(response.trials) == 2
        assert response.trials[0].title.startswith("Trial")
        assert response.trials[0].counts.met == 2
        assert timing.judged == 2
        assert timing.cache_hits == 0
        assert judge.calls == 2
        # The disclaimer travels in the payload so a frontend refactor cannot
        # drop it.
        assert "not medical advice" in response.disclaimer

    def test_a_second_identical_query_uses_the_cache(self, db: duckdb.DuckDBPyConnection) -> None:
        self._corpus(db)
        judge = _RecordingJudge()

        run_match(db=db, encoder=_FakeEncoder(), judge=judge, text=VIGNETTE, k=2)
        calls_after_first = judge.calls
        _, timing = run_match(db=db, encoder=_FakeEncoder(), judge=judge, text=VIGNETTE, k=2)

        assert timing.cache_hits == 2
        assert timing.judged == 0
        assert judge.calls == calls_after_first  # no further calls

    def test_one_failing_candidate_does_not_lose_the_others(
        self, db: duckdb.DuckDBPyConnection
    ) -> None:
        # The defect that killed a twelve-hour extraction pass, here in a place a
        # user would see.
        self._corpus(db, trials=2)

        class Flaky:
            model_name = "fake/model"

            def __init__(self) -> None:
                self.calls = 0

            def judge(self, text, criteria):  # type: ignore[no-untyped-def]
                self.calls += 1
                if self.calls == 1:
                    raise OpenRouterError("provider exploded")
                return [
                    CriterionVerdict(
                        ordinal=o,
                        text=t,
                        kind=k,
                        is_exclusion=x,
                        verdict="met",
                        rationale="fine",
                    )
                    for o, t, k, x in criteria
                ]

        response, _ = run_match(db=db, encoder=_FakeEncoder(), judge=Flaky(), text=VIGNETTE, k=2)

        assert len(response.trials) == 2
        failed = [t for t in response.trials if all(v.verdict == "unclear" for v in t.criteria)]
        assert len(failed) == 1
        assert "failed" in failed[0].criteria[0].rationale.lower()
        # …and the other trial still produced real verdicts.
        assert any(t.counts.met == 2 for t in response.trials)

    def test_k_bounds_the_candidates(self, db: duckdb.DuckDBPyConnection) -> None:
        self._corpus(db, trials=2)
        response, _ = run_match(
            db=db, encoder=_FakeEncoder(), judge=_RecordingJudge(), text=VIGNETTE, k=1
        )
        assert len(response.trials) == 1

    def test_an_empty_corpus_returns_no_trials(self, db: duckdb.DuckDBPyConnection) -> None:
        response, timing = run_match(
            db=db, encoder=_FakeEncoder(), judge=_RecordingJudge(), text=VIGNETTE, k=5
        )
        assert response.trials == []
        assert timing.judged == 0
