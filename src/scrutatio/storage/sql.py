"""Databricks SQL over REST.

Free Edition is serverless-only and offers no `databricks-connect` path that
would let a local process open a Spark session. The Statement Execution API
does everything the pipeline needs — DDL, ``COPY INTO``, ``MERGE`` — over plain
HTTP, which also means ingestion runs identically from a laptop, from GitHub
Actions, or from a job inside the workspace.

Verified against the workspace on 2026-08-15: DBSQL 2026.20 on the Serverless
Starter Warehouse.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Final

import httpx

from scrutatio.config import Settings, get_settings

logger = logging.getLogger(__name__)

_TERMINAL_STATES: Final = frozenset({"SUCCEEDED", "FAILED", "CANCELED", "CLOSED"})
_POLL_INTERVAL_SECONDS: Final = 2.0


class SqlError(RuntimeError):
    """A statement failed, or the warehouse could not be reached."""


class SqlClient:
    """Executes SQL statements against a serverless SQL warehouse."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        warehouse_id: str | None = None,
        client: httpx.Client | None = None,
        poll_timeout_seconds: float = 600.0,
    ) -> None:
        self._settings = settings or get_settings()
        if not self._settings.databricks_configured:
            msg = "DATABRICKS_HOST and DATABRICKS_TOKEN must be set to run SQL"
            raise ValueError(msg)
        self._warehouse_id = warehouse_id
        self._poll_timeout = poll_timeout_seconds
        self._owns_client = client is None
        # Deliberately NOT request_timeout_seconds: the server holds the
        # connection for the full wait_timeout, so an equal client timeout races
        # it and loses the statement_id while the statement keeps running.
        self._client = client or httpx.Client(timeout=self._settings.sql_timeout_seconds)

    def __enter__(self) -> SqlClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @property
    def _host(self) -> str:
        return str(self._settings.databricks_host).rstrip("/")

    @property
    def _headers(self) -> dict[str, str]:
        token = self._settings.databricks_token
        assert token is not None  # guaranteed by databricks_configured
        return {"Authorization": f"Bearer {token.get_secret_value()}"}

    def warehouse_id(self) -> str:
        """The configured warehouse, or the first one the workspace exposes."""
        if self._warehouse_id:
            return self._warehouse_id

        response = self._client.get(f"{self._host}/api/2.0/sql/warehouses", headers=self._headers)
        if response.status_code != httpx.codes.OK:
            msg = f"Could not list SQL warehouses: HTTP {response.status_code}"
            raise SqlError(msg)

        warehouses = response.json().get("warehouses", [])
        if not warehouses:
            msg = "No SQL warehouse available in this workspace"
            raise SqlError(msg)

        self._warehouse_id = str(warehouses[0]["id"])
        logger.info("Using SQL warehouse %s", self._warehouse_id)
        return self._warehouse_id

    def execute(
        self,
        statement: str,
        *,
        parameters: list[dict[str, Any]] | None = None,
    ) -> list[list[Any]]:
        """Run a statement to completion and return its rows.

        Values must be passed via ``parameters`` rather than interpolated —
        trial text is arbitrary prose and would otherwise break the statement or
        worse.
        """
        payload: dict[str, Any] = {
            "warehouse_id": self.warehouse_id(),
            "statement": statement,
            "wait_timeout": f"{self._settings.sql_wait_timeout_seconds}s",
            "on_wait_timeout": "CONTINUE",
        }
        if parameters:
            payload["parameters"] = parameters

        response = self._client.post(
            f"{self._host}/api/2.0/sql/statements",
            headers=self._headers,
            json=payload,
        )
        if response.status_code != httpx.codes.OK:
            msg = f"Statement rejected: HTTP {response.status_code}: {response.text[:400]}"
            raise SqlError(msg)

        body = self._await_completion(response.json())
        return body.get("result", {}).get("data_array", []) or []

    def _await_completion(self, body: dict[str, Any]) -> dict[str, Any]:
        deadline = time.monotonic() + self._poll_timeout
        statement_id = body.get("statement_id")

        while True:
            state = body.get("status", {}).get("state")

            if state == "SUCCEEDED":
                return body
            if state in _TERMINAL_STATES:
                error = body.get("status", {}).get("error", {})
                detail = error.get("message", state)
                msg = f"Statement {statement_id} ended as {state}: {detail}"
                raise SqlError(msg)

            if time.monotonic() >= deadline:
                msg = f"Statement {statement_id} still {state} after {self._poll_timeout:.0f}s"
                raise SqlError(msg)

            time.sleep(_POLL_INTERVAL_SECONDS)
            response = self._client.get(
                f"{self._host}/api/2.0/sql/statements/{statement_id}",
                headers=self._headers,
            )
            if response.status_code != httpx.codes.OK:
                msg = f"Could not poll statement {statement_id}: HTTP {response.status_code}"
                raise SqlError(msg)
            body = response.json()

    def upload_file(self, volume_path: str, content: bytes) -> None:
        """Write bytes to a Unity Catalog volume.

        ``volume_path`` is the path below ``/Volumes``, e.g.
        ``workspace/scrutatio/landing/bronze.ndjson``.
        """
        response = self._client.put(
            f"{self._host}/api/2.0/fs/files/Volumes/{volume_path.lstrip('/')}",
            headers={**self._headers, "Content-Type": "application/octet-stream"},
            params={"overwrite": "true"},
            content=content,
        )
        if response.status_code not in (httpx.codes.OK, httpx.codes.NO_CONTENT):
            msg = (
                f"Upload to {volume_path} failed: HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )
            raise SqlError(msg)
