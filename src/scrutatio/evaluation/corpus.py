"""Fetching the judged pool into a database of its own.

**A separate file, deliberately.** The product corpus is 11,195 currently
recruiting oncology trials; the TREC pool is 26,162 trials from a 2021 snapshot,
most of them neither recruiting nor oncological. Mixing them would silently change
what ``/match`` searches over and what every corpus statistic means. They are two
datasets with two purposes, so they get two files.

The overlap is 141 trials — 0.5%. So this is almost entirely new work, and that is
the honest cost of having an answer.

Fetching is free and fast: ``filter.ids`` takes a comma-separated batch, and the
whole pool is under a hundred requests. Extraction and embedding are what cost
money, which is why the baseline runs first — BM25 needs only the text landed
here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scrutatio.clients.ctgov import CtGovClient, nct_id
from scrutatio.storage.bronze import BRONZE_TABLE, ensure_storage, write_bronze

if TYPE_CHECKING:
    from collections.abc import Sequence

    import duckdb

logger = logging.getLogger(__name__)

# 300 ids per request is measured to work end to end. Larger batches risk the URL
# length limit for a GET, and the whole pool is only ~90 requests at this size.
DEFAULT_BATCH = 300


def already_landed(db: duckdb.DuckDBPyConnection) -> set[str]:
    """NCT ids already in this database, so a partial fetch resumes."""
    ensure_storage(db)
    return {r[0] for r in db.execute(f"SELECT nct_id FROM {BRONZE_TABLE}").fetchall()}  # noqa: S608


def fetch_pool(
    db: duckdb.DuckDBPyConnection,
    ncts: Sequence[str],
    *,
    batch_size: int = 300,
    client: CtGovClient | None = None,
) -> dict[str, int]:
    """Land every judged trial that is not already here.

    Returns counts rather than raising on gaps: a 2021 benchmark can name a study
    that has since been withdrawn from the registry, and losing the evaluation over
    a handful of them would be the wrong trade. The number of missing trials is
    reported so it can be stated alongside any result — measured at 0 of 300 in a
    sample, but that is a sample.
    """
    ensure_storage(db)
    have = already_landed(db)
    todo = [n for n in ncts if n not in have]
    logger.info("Pool: %d judged, %d already here, %d to fetch", len(ncts), len(have), len(todo))

    owns = client is None
    ctgov = client or CtGovClient()
    written = 0
    missing: list[str] = []

    try:
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            studies = list(ctgov.iter_studies(query={"filter.ids": ",".join(batch)}))
            written += write_bronze(db, studies, batch=f"trec-{start // batch_size}")

            returned = {nct_id(s) for s in studies}
            missing.extend(n for n in batch if n not in returned)
            logger.info(
                "Batch %d: asked %d, got %d (%d/%d done)",
                start // batch_size,
                len(batch),
                len(studies),
                start + len(batch),
                len(todo),
            )
    finally:
        if owns:
            ctgov.close()

    if missing:
        # Not an error, but it bounds the evaluation: a trial that cannot be
        # fetched can never be retrieved, so it counts against recall for both the
        # baseline and us — equally, which keeps the comparison fair.
        logger.warning("%d judged trials could not be fetched, e.g. %s", len(missing), missing[:5])

    return {"requested": len(ncts), "written": written, "missing": len(missing)}
