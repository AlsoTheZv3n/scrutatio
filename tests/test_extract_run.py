"""Extraction-run tests.

The run must survive being stopped and restarted, and must recognise the
difference between "no work left" and "the endpoint stopped answering". Getting
that distinction wrong means either a run that halts early on a healthy quota,
or one that burns its remaining attempts against a closed door.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scrutatio.extraction.runner import ExtractionOutcome, RunStats
from scrutatio.extraction.schema import Criterion, ExtractedEligibility
from scrutatio.pipeline.extract import ExtractionRunResult, run_extraction


def _outcome(nct: str) -> ExtractionOutcome:
    return ExtractionOutcome(
        nct_id=nct,
        result=ExtractedEligibility(
            criteria=[Criterion(criterion_id="c-0", text="t", kind="lab", is_exclusion=False)]
        ),
    )


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    stubs = {
        "ensure_silver": MagicMock(),
        "extraction_signature": MagicMock(return_value="sig0"),
        "pending_trials": MagicMock(return_value=[]),
        "write_silver": MagicMock(return_value=(0, 0)),
        "silver_stats": MagicMock(
            return_value={"trials": 0, "criteria": 0, "failed": 0, "total": 11200}
        ),
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(f"scrutatio.pipeline.extract.{name}", stub)
    return stubs


@pytest.fixture
def extractor() -> MagicMock:
    return MagicMock()


def _run(**kwargs: object) -> ExtractionRunResult:
    return run_extraction(run_token="t0", **kwargs)  # type: ignore[arg-type]


class TestLoop:
    def test_stops_when_nothing_is_pending(
        self, extractor: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        result = _run(db=MagicMock(), extractor=extractor)

        assert result.trials_written == 0
        assert patched["write_silver"].call_count == 0

    def test_storage_is_ensured_before_any_work(
        self, extractor: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        _run(db=MagicMock(), extractor=extractor)
        patched["ensure_silver"].assert_called_once()

    def test_commits_each_batch_before_fetching_the_next(
        self, monkeypatch: pytest.MonkeyPatch, extractor: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        # An interrupted run must keep everything it finished.
        patched["pending_trials"].side_effect = [
            [("NCT00000001", "text")],
            [("NCT00000002", "text")],
            [],
        ]
        patched["write_silver"].return_value = (1, 1)
        monkeypatch.setattr(
            "scrutatio.pipeline.extract.extract_many",
            lambda _e, items, **_k: iter([_outcome(n) for n, _ in items]),
        )

        result = _run(db=MagicMock(), extractor=extractor, batch_size=1)

        assert result.trials_written == 2
        assert patched["write_silver"].call_count == 2

    def test_batch_names_are_unique_within_a_run(
        self, monkeypatch: pytest.MonkeyPatch, extractor: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        patched["pending_trials"].side_effect = [
            [("NCT00000001", "t")],
            [("NCT00000002", "t")],
            [],
        ]
        patched["write_silver"].return_value = (1, 1)
        monkeypatch.setattr(
            "scrutatio.pipeline.extract.extract_many",
            lambda _e, items, **_k: iter([_outcome(n) for n, _ in items]),
        )

        _run(db=MagicMock(), extractor=extractor, batch_size=1)

        names = [c.kwargs["batch"] for c in patched["write_silver"].call_args_list]
        assert len(set(names)) == len(names)

    def test_limit_bounds_the_work(
        self, monkeypatch: pytest.MonkeyPatch, extractor: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        patched["pending_trials"].return_value = [("NCT00000001", "t")]
        patched["write_silver"].return_value = (1, 1)
        monkeypatch.setattr(
            "scrutatio.pipeline.extract.extract_many",
            lambda _e, items, **_k: iter([_outcome(n) for n, _ in items]),
        )

        result = _run(db=MagicMock(), extractor=extractor, limit=3, batch_size=10)

        assert result.trials_written == 3
        # The final page request must never exceed the remaining budget.
        assert all(c.kwargs["limit"] <= 3 for c in patched["pending_trials"].call_args_list)


class TestThrottling:
    def test_stops_when_a_whole_batch_fails_under_throttling(
        self, monkeypatch: pytest.MonkeyPatch, extractor: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        # Continuing here just burns the remaining attempts against a closed door.
        patched["pending_trials"].return_value = [("NCT00000001", "t")]
        patched["write_silver"].return_value = (0, 0)

        def throttled(_e: object, items: object, **kwargs: object) -> object:
            stats = kwargs["stats"]
            assert isinstance(stats, RunStats)
            stats.record(ok=False, throttled=True)
            return iter([ExtractionOutcome(nct_id="NCT00000001", result=None, error="HTTP 429")])

        monkeypatch.setattr("scrutatio.pipeline.extract.extract_many", throttled)

        result = _run(db=MagicMock(), extractor=extractor, batch_size=1)

        assert patched["write_silver"].call_count == 1
        assert result.rate_limited == 1

    def test_an_empty_batch_without_throttling_just_ends_the_run(
        self, monkeypatch: pytest.MonkeyPatch, extractor: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        patched["pending_trials"].side_effect = [[("NCT00000001", "t")], []]
        patched["write_silver"].return_value = (0, 0)
        monkeypatch.setattr(
            "scrutatio.pipeline.extract.extract_many", lambda _e, _i, **_k: iter([])
        )

        result = _run(db=MagicMock(), extractor=extractor, batch_size=1)
        assert result.rate_limited == 0


class TestResultReporting:
    def test_remaining_comes_from_the_table_not_the_counter(
        self, extractor: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        patched["silver_stats"].return_value = {
            "trials": 4000,
            "criteria": 120000,
            "failed": 3,
            "total": 11200,
        }
        assert _run(db=MagicMock(), extractor=extractor).remaining == 7200

    def test_quota_exhausted_needs_work_left_and_throttling(self) -> None:
        stopped = ExtractionRunResult(
            signature="s",
            trials_written=10,
            criteria_written=30,
            succeeded=10,
            failed=5,
            rate_limited=5,
            remaining=9000,
        )
        assert stopped.quota_exhausted is True

    def test_a_finished_run_is_not_quota_exhausted(self) -> None:
        done = ExtractionRunResult(
            signature="s",
            trials_written=11200,
            criteria_written=300000,
            succeeded=11200,
            failed=0,
            rate_limited=0,
            remaining=0,
        )
        assert done.quota_exhausted is False

    def test_failures_without_throttling_are_not_quota_exhausted(self) -> None:
        # Schema failures are a different problem and need a different response.
        broken = ExtractionRunResult(
            signature="s",
            trials_written=5,
            criteria_written=10,
            succeeded=5,
            failed=5,
            rate_limited=0,
            remaining=9000,
        )
        assert broken.quota_exhausted is False


class TestResilience:
    """One trial must never end the run.

    A JSONDecodeError from a malformed response body escaped a worker, passed
    through future.result(), and killed a pass that was three batches in. Batch
    commits meant nothing was lost, but 10,000 trials of remaining work stopped.
    """

    def test_an_unexpected_exception_becomes_a_failed_outcome(self) -> None:
        from scrutatio.extraction.runner import extract_many

        class Exploding:
            def extract(self, text: str) -> object:
                if "boom" in text:
                    raise ValueError("malformed response body")
                return SimpleNamespace(criteria=[])

        outcomes = list(
            extract_many(
                Exploding(),  # type: ignore[arg-type]
                [("NCT00000001", "fine"), ("NCT00000002", "boom"), ("NCT00000003", "fine")],
                max_workers=1,
            )
        )

        assert len(outcomes) == 3
        by_id = {o.nct_id: o for o in outcomes}
        assert by_id["NCT00000002"].ok is False
        assert "ValueError" in (by_id["NCT00000002"].error or "")
        # The trials either side of the explosion still produced results.
        assert by_id["NCT00000001"].ok is True
        assert by_id["NCT00000003"].ok is True
