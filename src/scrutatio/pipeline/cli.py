"""Command line entry point.

`uv run scrutatio --help`
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime

from scrutatio.config import get_settings
from scrutatio.ctgov import CtGovClient
from scrutatio.extraction import EligibilityExtractor
from scrutatio.pipeline.backfill import run_backfill
from scrutatio.pipeline.extract import run_extraction
from scrutatio.storage import (
    BRONZE_TABLE,
    bronze_count,
    bronze_max_update,
    database,
    ensure_silver,
    ensure_storage,
    extraction_signature,
    safe_watermark,
    silver_stats,
)
from scrutatio.storage.db import DatabaseBusyError


def _build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="scrutatio",
        description="Oncology clinical-trial matching — pipeline commands.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log every step")
    sub = parser.add_subparsers(dest="command", required=True)

    backfill = sub.add_parser("backfill", help="land ClinicalTrials.gov studies into Bronze")
    backfill.add_argument(
        "--incremental",
        action="store_true",
        help="fetch only studies updated since the last completed run",
    )
    backfill.add_argument("--limit", type=int, help="stop after N studies (for smoke runs)")
    backfill.add_argument("--batch-size", type=int, default=500, help="studies per batch")

    extract = sub.add_parser("extract", help="extract eligibility criteria into Silver")
    extract.add_argument("--limit", type=int, help="stop after N trials")
    extract.add_argument("--batch-size", type=int, default=100, help="trials committed per batch")
    extract.add_argument(
        "--workers",
        type=int,
        default=settings.extraction_workers,
        help="concurrent extraction calls",
    )
    extract.add_argument(
        "--retry-failed",
        action="store_true",
        help="include trials that already exhausted their attempts",
    )

    sub.add_parser("status", help="show pipeline progress")

    return parser


def _cmd_backfill(args: argparse.Namespace) -> int:
    run_token = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    with database() as db, CtGovClient() as ctgov:
        result = run_backfill(
            db=db,
            ctgov=ctgov,
            incremental=args.incremental,
            limit=args.limit,
            batch_size=args.batch_size,
            run_token=run_token,
        )

    mode = f"incremental from {result.incremental_from}" if result.incremental_from else "full"
    print(f"Backfill ({mode})")
    print(f"  batches written   {result.batches}")
    print(f"  studies landed    {result.written}")
    print(f"  rows before/after {result.rows_before} -> {result.rows_after}")
    print(f"  net new           {result.added}")
    if not result.complete:
        print("  NOTE: partial run — watermark not advanced, next incremental starts earlier")
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    run_token = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    with database() as db, EligibilityExtractor() as extractor:
        result = run_extraction(
            db=db,
            extractor=extractor,
            limit=args.limit,
            batch_size=args.batch_size,
            workers=args.workers,
            retry_failed=args.retry_failed,
            run_token=run_token,
        )
        usage = extractor.usage

    print("Extraction")
    print(f"  signature        {result.signature}")
    print(f"  trials written   {result.trials_written}")
    print(f"  criteria written {result.criteria_written}")
    print(f"  succeeded/failed {result.succeeded}/{result.failed}")
    print(f"  rate limited     {result.rate_limited}")
    print(f"  still pending    {result.remaining}")
    if usage.calls:
        print(f"  tokens in/out    {usage.prompt_tokens}/{usage.completion_tokens}")
        # >> 1.0 means output tokens were spent on something other than the
        # answer — the reasoning-model failure that cost a previous full run.
        print(f"  reasoning ratio  {usage.reasoning_ratio:.2f}  (near 1.0 is healthy)")
        if usage.wasted_calls:
            # Truncated attempts burn a full budget and return nothing. Counted
            # separately because they are invisible in the criteria totals.
            print(f"  wasted calls     {usage.wasted_calls}/{usage.calls} (truncated)")
    if result.quota_exhausted:
        print("  NOTE: stopped on throttling, not on work — rerun later to continue")
    return 0


def _cmd_status(_: argparse.Namespace) -> int:
    with database() as db:
        # Status must work on a fresh database, before any run has created tables.
        ensure_storage(db)
        ensure_silver(db)
        signature = extraction_signature()
        silver = silver_stats(db, signature)
        done = silver["trials"]
        total = silver["total"] or 1
        print("Bronze")
        print(f"  table          {BRONZE_TABLE}")
        print(f"  rows           {bronze_count(db)}")
        print(f"  newest update  {bronze_max_update(db) or '-'}")
        print(f"  resume point   {safe_watermark(db) or '- (no completed run; next pass is full)'}")
        print("Silver")
        print(f"  signature      {signature}")
        print(f"  trials         {done}/{silver['total']}  ({100 * done / total:.1f}%)")
        print(f"  criteria       {silver['criteria']}")
        print(f"  failed         {silver['failed']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    handlers = {"backfill": _cmd_backfill, "extract": _cmd_extract, "status": _cmd_status}
    try:
        return handlers[args.command](args)
    except DatabaseBusyError as exc:
        # Expected while another run holds the file; a traceback would suggest a
        # defect rather than "wait a moment".
        print(f"Database busy: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
