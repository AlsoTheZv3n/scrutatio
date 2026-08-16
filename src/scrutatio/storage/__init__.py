"""Local persistence in a single DuckDB file."""

from scrutatio.storage.bronze import (
    BRONZE_TABLE,
    DEFAULT_BATCH_SIZE,
    RUNS_TABLE,
    bronze_count,
    bronze_max_update,
    ensure_storage,
    finish_run,
    safe_watermark,
    start_run,
    write_bronze,
)
from scrutatio.storage.db import IN_MEMORY, connect, database
from scrutatio.storage.silver import (
    FAILURES_TABLE,
    SILVER_TABLE,
    ensure_silver,
    extraction_signature,
    pending_trials,
    silver_stats,
    write_silver,
)

__all__ = [
    "BRONZE_TABLE",
    "DEFAULT_BATCH_SIZE",
    "FAILURES_TABLE",
    "IN_MEMORY",
    "RUNS_TABLE",
    "SILVER_TABLE",
    "bronze_count",
    "bronze_max_update",
    "connect",
    "database",
    "ensure_silver",
    "ensure_storage",
    "extraction_signature",
    "finish_run",
    "pending_trials",
    "safe_watermark",
    "silver_stats",
    "start_run",
    "write_bronze",
    "write_silver",
]
