"""Backfill tests.

The behaviour worth pinning down: batching splits correctly, incremental mode
reads its starting point from Bronze rather than guessing, and an empty Bronze
degrades to a full pass instead of fetching nothing.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import MagicMock

import pytest

from scrutatio.clients.ctgov.models import Study
from scrutatio.pipeline import run_backfill
from scrutatio.pipeline.backfill import _chunks
from scrutatio.pipeline.cli import main


def _study(nct: str) -> Study:
    return Study.model_validate(
        {
            "protocolSection": {
                "identificationModule": {"nctId": nct},
                "statusModule": {
                    "overallStatus": "RECRUITING",
                    "lastUpdatePostDateStruct": {"date": "2026-08-14", "type": "ACTUAL"},
                },
            }
        }
    )


@pytest.fixture
def db() -> MagicMock:
    """The storage layer is stubbed here; its real behaviour is tested against a
    real DuckDB in test_duck_storage.py."""
    return MagicMock()


@pytest.fixture
def ctgov() -> MagicMock:
    client = MagicMock()
    client.iter_studies.return_value = iter([_study(f"NCT{i:08d}") for i in range(1, 6)])
    return client


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    """Stub out the storage layer; its own behaviour is tested elsewhere."""
    stubs = {
        "ensure_storage": MagicMock(),
        "write_bronze": MagicMock(side_effect=lambda _s, chunk, batch: len(list(chunk))),
        "bronze_count": MagicMock(side_effect=[0, 5]),
        "safe_watermark": MagicMock(return_value=None),
        "start_run": MagicMock(),
        "finish_run": MagicMock(),
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(f"scrutatio.pipeline.backfill.{name}", stub)
    return stubs


class TestChunking:
    def test_splits_evenly(self) -> None:
        assert [len(c) for c in _chunks(range(10), 4)] == [4, 4, 2]  # type: ignore[arg-type]

    def test_empty_input_yields_nothing(self) -> None:
        assert list(_chunks([], 4)) == []

    def test_batch_larger_than_input(self) -> None:
        assert [len(c) for c in _chunks(range(3), 100)] == [3]  # type: ignore[arg-type]


class TestRunBackfill:
    def test_writes_in_batches(
        self, db: MagicMock, ctgov: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        result = run_backfill(run_token="t0", db=db, ctgov=ctgov, batch_size=2)

        assert result.written == 5
        assert result.batches == 3
        assert patched["write_bronze"].call_count == 3

    def test_batch_names_are_unique_and_ordered(
        self, db: MagicMock, ctgov: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        run_backfill(run_token="t0", db=db, ctgov=ctgov, batch_size=2)

        names = [c.kwargs["batch"] for c in patched["write_bronze"].call_args_list]
        assert names == ["backfill-t0-0000", "backfill-t0-0001", "backfill-t0-0002"]

    def test_batch_names_differ_between_runs(
        self, db: MagicMock, ctgov: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        # Two overlapping runs sharing a batch name would overwrite each other's
        # landing file before COPY INTO reads it.
        run_backfill(run_token="aaa", db=db, ctgov=ctgov, batch_size=10)
        first = patched["write_bronze"].call_args_list[-1].kwargs["batch"]

        ctgov.iter_studies.return_value = iter([_study("NCT00000009")])
        patched["bronze_count"].side_effect = [5, 6]
        run_backfill(run_token="bbb", db=db, ctgov=ctgov, batch_size=10)
        second = patched["write_bronze"].call_args_list[-1].kwargs["batch"]

        assert first != second


class TestRunCompletion:
    """A partial run must not publish a resume point. Otherwise the next
    incremental pass starts after studies that never actually landed.
    """

    def test_full_pass_is_marked_complete(
        self, db: MagicMock, ctgov: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        result = run_backfill(run_token="t0", db=db, ctgov=ctgov)

        assert result.complete is True
        assert patched["finish_run"].call_args.kwargs["complete"] is True

    def test_limited_run_is_never_complete(
        self, db: MagicMock, ctgov: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        # --limit walks part of the scope; its endpoint is not a resume point.
        result = run_backfill(run_token="t0", db=db, ctgov=ctgov, limit=3)

        assert result.complete is False
        assert patched["finish_run"].call_args.kwargs["complete"] is False

    def test_failure_mid_run_still_records_an_incomplete_run(
        self, db: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        ctgov = MagicMock()
        ctgov.iter_studies.return_value = iter([_study("NCT00000001")])
        patched["write_bronze"].side_effect = RuntimeError("warehouse died")

        with pytest.raises(RuntimeError, match="warehouse died"):
            run_backfill(run_token="t0", db=db, ctgov=ctgov)

        patched["finish_run"].assert_called_once()
        assert patched["finish_run"].call_args.kwargs["complete"] is False

    def test_run_is_registered_before_any_write(
        self, db: MagicMock, ctgov: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        run_backfill(run_token="t0", db=db, ctgov=ctgov)
        patched["start_run"].assert_called_once()
        assert patched["start_run"].call_args[0][1] == "backfill-t0"

    def test_reports_net_new_separately_from_written(
        self, db: MagicMock, ctgov: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        # Rerunning writes rows that are updates, not additions.
        patched["bronze_count"].side_effect = [5, 5]
        result = run_backfill(run_token="t0", db=db, ctgov=ctgov, batch_size=10)

        assert result.written == 5
        assert result.added == 0

    def test_storage_is_ensured_first(
        self, db: MagicMock, ctgov: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        run_backfill(run_token="t0", db=db, ctgov=ctgov)
        patched["ensure_storage"].assert_called_once_with(db)

    def test_full_pass_does_not_filter_by_date(
        self, db: MagicMock, ctgov: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        run_backfill(run_token="t0", db=db, ctgov=ctgov)
        assert ctgov.iter_studies.call_args.kwargs["updated_since"] is None

    def test_incremental_starts_from_the_bronze_watermark(
        self, db: MagicMock, ctgov: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        patched["safe_watermark"].return_value = date(2026, 8, 1)

        result = run_backfill(run_token="t0", db=db, ctgov=ctgov, incremental=True)

        assert ctgov.iter_studies.call_args.kwargs["updated_since"] == date(2026, 8, 1)
        assert result.incremental_from == date(2026, 8, 1)

    def test_incremental_on_empty_bronze_falls_back_to_full(
        self, db: MagicMock, ctgov: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        # Otherwise the first scheduled run would silently fetch nothing.
        patched["safe_watermark"].return_value = None

        result = run_backfill(run_token="t0", db=db, ctgov=ctgov, incremental=True)

        assert ctgov.iter_studies.call_args.kwargs["updated_since"] is None
        assert result.incremental_from is None

    def test_limit_is_passed_through(
        self, db: MagicMock, ctgov: MagicMock, patched: dict[str, MagicMock]
    ) -> None:
        run_backfill(run_token="t0", db=db, ctgov=ctgov, limit=7)
        assert ctgov.iter_studies.call_args.kwargs["limit"] == 7

    def test_empty_scope_writes_nothing(self, db: MagicMock, patched: dict[str, MagicMock]) -> None:
        ctgov = MagicMock()
        ctgov.iter_studies.return_value = iter([])
        patched["bronze_count"].side_effect = [0, 0]

        result = run_backfill(run_token="t0", db=db, ctgov=ctgov)

        assert (result.written, result.batches) == (0, 0)
        assert patched["write_bronze"].call_count == 0


class TestCli:
    def test_status_prints_table_and_count(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("scrutatio.pipeline.cli.database", MagicMock())
        monkeypatch.setattr("scrutatio.pipeline.cli.ensure_storage", MagicMock())
        monkeypatch.setattr("scrutatio.pipeline.cli.ensure_silver", MagicMock())
        monkeypatch.setattr(
            "scrutatio.pipeline.cli.silver_stats",
            lambda _s, _sig: {"trials": 3000, "criteria": 90000, "failed": 12, "total": 11200},
        )
        monkeypatch.setattr("scrutatio.pipeline.cli.bronze_count", lambda _: 11200)
        monkeypatch.setattr("scrutatio.pipeline.cli.bronze_max_update", lambda _: "2026-08-14")
        monkeypatch.setattr("scrutatio.pipeline.cli.safe_watermark", lambda _: "2026-08-01")

        assert main(["status"]) == 0
        out = capsys.readouterr().out
        assert "11200" in out
        assert "bronze_studies" in out
        # The two dates are different things and must be shown separately.
        assert "2026-08-14" in out
        assert "2026-08-01" in out

    def test_status_says_when_no_resume_point_exists(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("scrutatio.pipeline.cli.database", MagicMock())
        monkeypatch.setattr("scrutatio.pipeline.cli.ensure_storage", MagicMock())
        monkeypatch.setattr("scrutatio.pipeline.cli.ensure_silver", MagicMock())
        monkeypatch.setattr(
            "scrutatio.pipeline.cli.silver_stats",
            lambda _s, _sig: {"trials": 0, "criteria": 0, "failed": 0, "total": 6000},
        )
        monkeypatch.setattr("scrutatio.pipeline.cli.bronze_count", lambda _: 6000)
        monkeypatch.setattr("scrutatio.pipeline.cli.bronze_max_update", lambda _: "2026-08-14")
        monkeypatch.setattr("scrutatio.pipeline.cli.safe_watermark", lambda _: None)

        main(["status"])
        assert "next pass is full" in capsys.readouterr().out

    def test_backfill_reports_counts(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("scrutatio.pipeline.cli.database", MagicMock())
        monkeypatch.setattr("scrutatio.pipeline.cli.CtGovClient", MagicMock())

        def fake_run(**_: Any) -> Any:
            from scrutatio.pipeline import BackfillResult

            return BackfillResult(
                written=200,
                batches=1,
                rows_before=0,
                rows_after=200,
                incremental_from=None,
                complete=True,
            )

        monkeypatch.setattr("scrutatio.pipeline.cli.run_backfill", fake_run)

        assert main(["backfill"]) == 0
        assert "studies landed    200" in capsys.readouterr().out

    def test_backfill_warns_when_the_run_was_partial(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("scrutatio.pipeline.cli.database", MagicMock())
        monkeypatch.setattr("scrutatio.pipeline.cli.CtGovClient", MagicMock())

        def fake_run(**_: Any) -> Any:
            from scrutatio.pipeline import BackfillResult

            return BackfillResult(
                written=5000,
                batches=10,
                rows_before=0,
                rows_after=5000,
                incremental_from=None,
                complete=False,
            )

        monkeypatch.setattr("scrutatio.pipeline.cli.run_backfill", fake_run)

        main(["backfill", "--limit", "5000"])
        assert "watermark not advanced" in capsys.readouterr().out

    def test_command_is_required(self) -> None:
        with pytest.raises(SystemExit):
            main([])
