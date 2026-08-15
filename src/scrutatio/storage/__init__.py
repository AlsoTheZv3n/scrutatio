"""Delta persistence in Unity Catalog, driven over the SQL REST API."""

from scrutatio.storage.bronze import (
    BRONZE_TABLE,
    CATALOG,
    DEFAULT_BATCH_SIZE,
    RUNS_TABLE,
    SCHEMA,
    VOLUME,
    UnsafeBatchNameError,
    bronze_count,
    bronze_max_update,
    ensure_storage,
    finish_run,
    safe_watermark,
    start_run,
    write_bronze,
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
from scrutatio.storage.sql import SqlClient, SqlError

__all__ = [
    "BRONZE_TABLE",
    "CATALOG",
    "DEFAULT_BATCH_SIZE",
    "FAILURES_TABLE",
    "RUNS_TABLE",
    "SCHEMA",
    "SILVER_TABLE",
    "VOLUME",
    "SqlClient",
    "SqlError",
    "UnsafeBatchNameError",
    "bronze_count",
    "bronze_max_update",
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
