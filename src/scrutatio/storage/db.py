"""The database: one DuckDB file.

Replaces the Databricks Statement Execution API. The whole corpus is 11,200
studies and ~300,000 criteria — a scale at which a single embedded file is not a
compromise but the correct answer. It also removes an entire class of failure:
there is no warehouse to be unavailable, no catalog name that differs per
workspace, and no statement held server-side past a client timeout.

Vector search comes with it. DuckDB has `FLOAT[N]` columns and a native
`array_cosine_similarity`, so Gold is a table and top-K is a query rather than a
managed index with a sync mechanism.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from scrutatio.config import get_settings

logger = logging.getLogger(__name__)

# `:memory:` is DuckDB's in-process database; tests use it so they exercise real
# SQL instead of a recording of an HTTP API.
IN_MEMORY = ":memory:"

# DuckDB appends this to the (localised) OS error when the file is locked.
_LOCK_MARKER = "File is already open"


class DatabaseBusyError(RuntimeError):
    """Another process holds the database.

    DuckDB allows a single read-write process and excludes every other
    connection while it is attached — a read-only connection is refused too,
    which is measured rather than assumed. That is workable for a batch
    pipeline, but it means progress cannot be inspected mid-run and anything
    long-lived that wants to read (a UI) needs an exported snapshot rather than
    the live file.
    """


def connect(path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Open the database, creating the parent directory if needed.

    Pass ``:memory:`` for an ephemeral database.
    """
    target = path if path is not None else get_settings().db_path
    if target != IN_MEMORY:
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target = str(target)
    logger.info("Opening DuckDB at %s", target)
    try:
        return duckdb.connect(target)
    except duckdb.IOException as exc:
        # Match DuckDB's own wording, not the OS error it wraps: that half of the
        # message is localised and would not match on a non-English system.
        if _LOCK_MARKER not in str(exc):
            raise
        msg = (
            f"{target} is held by another process — DuckDB allows only one at a time. "
            f"A backfill or extraction is probably still running; wait for it to finish."
        )
        raise DatabaseBusyError(msg) from exc


@contextmanager
def database(path: Path | str | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open the database for the duration of a block, then close it."""
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()
