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
from scrutatio.storage.db import IN_MEMORY, DatabaseBusyError, connect, database
from scrutatio.storage.gold import (
    GOLD_TABLE,
    ensure_gold,
    gold_stats,
    pending_criteria,
    search_criteria,
    write_embeddings,
)
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
    "GOLD_TABLE",
    "IN_MEMORY",
    "RUNS_TABLE",
    "SILVER_TABLE",
    "DatabaseBusyError",
    "bronze_count",
    "bronze_max_update",
    "connect",
    "database",
    "ensure_gold",
    "ensure_silver",
    "ensure_storage",
    "extraction_signature",
    "finish_run",
    "gold_stats",
    "pending_criteria",
    "pending_trials",
    "safe_watermark",
    "search_criteria",
    "silver_stats",
    "start_run",
    "write_bronze",
    "write_embeddings",
    "write_silver",
]
