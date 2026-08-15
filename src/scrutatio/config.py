"""Typed configuration, loaded from environment or `.env`.

Every value that varies between machines lives here. Nothing else reads
`os.environ` directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, HttpUrl, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Chat endpoints measured as available on this workspace (2026-08-15).
# qwen3-next is the default: it returns `content` as a plain string, splits
# condition from biomarker correctly, and spends ~143 completion tokens where
# the gpt-oss models spend 347-460 on reasoning for the same result.
ChatEndpoint = Literal[
    "databricks-qwen3-next-80b-a3b-instruct",
    "databricks-qwen35-122b-a10b",
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-meta-llama-3-1-8b-instruct",
    "databricks-llama-4-maverick",
    "databricks-gemma-3-12b",
    "databricks-gpt-oss-120b",
    "databricks-gpt-oss-20b",
]

# All three measured at 1024 dimensions. gte is the default over bge because
# bge-large-en truncates at 512 tokens and eligibility texts average ~705.
EmbeddingEndpoint = Literal[
    "databricks-gte-large-en",
    "databricks-bge-large-en",
    "databricks-qwen3-embedding-0-6b",
]

EMBEDDING_DIMENSIONS = 1024

# Interventional, recruiting, cancer by MeSH ancestor term. Measured 2026-08-15:
# 11_200 studies. The looser `query.cond=cancer OR neoplasm` yields 18_821 and
# pulls in observational registries.
DEFAULT_SCOPE_FILTER = "AREA[ConditionAncestorTerm]Neoplasms AND AREA[StudyType]INTERVENTIONAL"


class Settings(BaseSettings):
    """Application settings.

    Reads `.env` if present; real environment variables win over file values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Databricks -------------------------------------------------------
    databricks_host: HttpUrl | None = Field(
        default=None,
        description="Workspace URL, e.g. https://dbc-xxxxxxxx-xxxx.cloud.databricks.com",
    )
    databricks_token: SecretStr | None = Field(
        default=None,
        description="Personal access token (dapi...). Never logged, never committed.",
    )

    # --- Models -----------------------------------------------------------
    chat_endpoint: ChatEndpoint = "databricks-qwen3-next-80b-a3b-instruct"
    embedding_endpoint: EmbeddingEndpoint = "databricks-gte-large-en"

    # --- ClinicalTrials.gov ----------------------------------------------
    ctgov_base_url: HttpUrl = HttpUrl("https://clinicaltrials.gov/api/v2")
    scope_filter: str = DEFAULT_SCOPE_FILTER
    recruiting_only: bool = True
    page_size: int = Field(default=1000, ge=1, le=1000)

    # --- Behaviour --------------------------------------------------------
    # Three separate budgets, because the three services have nothing in common.
    # A single shared value was a latent bug: LLM extraction was measured at
    # 25.6s for one call on a real trial, leaving ~4s of headroom against a 30s
    # ceiling, and SQL statements are held server-side for the full wait_timeout.
    request_timeout_seconds: float = Field(
        default=30.0, gt=0, description="ClinicalTrials.gov HTTP timeout."
    )
    sql_timeout_seconds: float = Field(
        default=90.0,
        gt=0,
        description=(
            "Databricks SQL HTTP timeout. Must exceed the server-side wait_timeout, "
            "or the connection drops while the statement is still running and its "
            "statement_id is lost — the statement keeps executing, unreachable."
        ),
    )
    llm_timeout_seconds: float = Field(
        default=180.0,
        gt=0,
        description="Serving-endpoint timeout. Criteria-heavy trials take 25s+ per call.",
    )
    max_retries: int = Field(default=5, ge=0)

    # Held server-side before the API returns and polling takes over.
    sql_wait_timeout_seconds: int = Field(default=30, ge=5, le=50)

    @field_validator("scope_filter")
    @classmethod
    def _scope_filter_not_blank(cls, v: str) -> str:
        if not v.strip():
            msg = "scope_filter must not be empty; it bounds the entire corpus"
            raise ValueError(msg)
        return v.strip()

    @model_validator(mode="after")
    def _sql_timeout_exceeds_wait(self) -> Settings:
        if self.sql_timeout_seconds <= self.sql_wait_timeout_seconds:
            msg = (
                f"sql_timeout_seconds ({self.sql_timeout_seconds}) must exceed "
                f"sql_wait_timeout_seconds ({self.sql_wait_timeout_seconds}); otherwise the "
                f"client hangs up while the server still holds the statement"
            )
            raise ValueError(msg)
        return self

    @property
    def databricks_configured(self) -> bool:
        """True when both host and token are present."""
        return self.databricks_host is not None and self.databricks_token is not None

    def serving_endpoint_url(self, endpoint: str) -> str:
        """Full invocation URL for a Databricks serving endpoint."""
        if self.databricks_host is None:
            msg = "databricks_host is not configured"
            raise ValueError(msg)
        host = str(self.databricks_host).rstrip("/")
        return f"{host}/serving-endpoints/{endpoint}/invocations"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
