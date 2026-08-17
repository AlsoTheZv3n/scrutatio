"""Typed configuration, loaded from environment or `.env`.

Every value that varies between machines lives here. Nothing else reads
`os.environ` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Extraction and matching both go through OpenRouter, which speaks the OpenAI
# chat-completions shape. The model is a plain string rather than a Literal: the
# catalogue held 413 entries when it was last read (318 of them supporting
# structured outputs), so an allow-list would be stale within the week.
DEFAULT_EXTRACTION_MODEL = "deepseek/deepseek-v4-flash"

# Embeddings run locally — no API, no per-vector cost.
#
# The original choice was gte-large-en-v1.5 for its 8192-token window, because a
# whole eligibility text averages ~835 tokens and BGE truncates at 512. That
# reasoning does not apply here: we embed *criteria*, not whole texts, and a
# criterion measures ~125 characters — about 31 tokens. The window is 16x more
# than needed either way, so it stops being a differentiator.
#
# BGE wins on what is left: it needs no `trust_remote_code=True`, so building the
# index does not execute code fetched from a model repository. Same 1024
# dimensions, so nothing downstream changes.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSIONS = 1024

# BGE is trained with an instruction prefix on the *query* side only; documents
# are embedded bare. Omitting it costs retrieval quality, and applying it to both
# sides costs more.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Interventional, recruiting, cancer by MeSH ancestor term. Measured 2026-08-16:
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

    # --- OpenRouter -------------------------------------------------------
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        description="OpenRouter API key. Never logged, never committed.",
    )
    openrouter_base_url: HttpUrl = HttpUrl("https://openrouter.ai/api/v1")
    extraction_model: str = DEFAULT_EXTRACTION_MODEL
    matching_model: str = DEFAULT_EXTRACTION_MODEL

    # Reasoning is waste for structured extraction: the schema already fixes the
    # shape of the answer, so thinking tokens buy nothing and are billed and
    # timed like any other output. Measured on two real trials — disabling it cut
    # output tokens 2.5-3.8x, moved the completion-to-answer ratio from ~3.5 to
    # ~1.0, and left the criterion count unchanged.
    #
    # This is the same diagnosis the Databricks post-mortem reached about
    # gpt-oss-120b, except here it is a request parameter rather than a reason to
    # change models. Matching may want it on; extraction does not.
    extraction_reasoning: bool = False

    # 18 providers serve deepseek-v4-flash, and default routing scatters requests
    # across them. Measured on four real trials, interleaved: default routing
    # ranged 46.8-112.7 tok/s (a 2.4x spread, hitting Parasail, Alibaba,
    # AtlasCloud and StreamLake); `sort=throughput` pinned to one fast provider
    # and ranged 73.7-111.7.
    #
    # The median only improves 1.31x, but the median is not what costs time: a
    # batch finishes when the slowest of its concurrent requests does, so one
    # slow provider in the wave paces the whole batch. Overnight throughput fell
    # from ~950 to ~350 trials/hour with zero rate limiting, which is what that
    # looks like. Set to None to let OpenRouter choose.
    extraction_provider_sort: str | None = "throughput"

    # --- Local storage ----------------------------------------------------
    # One file. `data/` is gitignored.
    db_path: Path = Path("data/scrutatio.duckdb")

    # --- Embeddings -------------------------------------------------------
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    # --- Spend guard ------------------------------------------------------
    # The one ceiling under our own control: bulk runs check the projected spend
    # before starting and refuse to proceed past it. Sized against the measured
    # corpus (~16.8M input / ~18.3M output tokens for a full pass), not a guess.
    max_spend_usd: float = Field(
        default=25.0,
        gt=0,
        description="Refuse to start a bulk run whose projected cost exceeds this figure.",
    )

    # --- ClinicalTrials.gov ----------------------------------------------
    ctgov_base_url: HttpUrl = HttpUrl("https://clinicaltrials.gov/api/v2")
    scope_filter: str = DEFAULT_SCOPE_FILTER
    recruiting_only: bool = True
    page_size: int = Field(default=1000, ge=1, le=1000)

    # --- HTTP API ---------------------------------------------------------
    # The frontend is a Vite dev server on its own port, so every call is
    # cross-origin. Listed explicitly rather than "*": there is no reason for a
    # third-party page to be able to drive this.
    api_cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"],
        description="Origins allowed to call the API.",
    )
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)

    # --- Behaviour --------------------------------------------------------
    request_timeout_seconds: float = Field(
        default=30.0, gt=0, description="ClinicalTrials.gov HTTP timeout."
    )
    llm_timeout_seconds: float = Field(
        default=180.0,
        gt=0,
        description="OpenRouter timeout. Criteria-heavy trials take 25s+ per call.",
    )
    max_retries: int = Field(default=5, ge=0)

    # OpenRouter publishes no concurrency limit for paid accounts and reroutes
    # around provider throttling, so the safe number is unknown until measured.
    # Start low and raise on evidence — the previous platform was tuned downward
    # from 8 and lost a night to it.
    extraction_workers: int = Field(default=4, ge=1, le=64)

    @field_validator("scope_filter")
    @classmethod
    def _scope_filter_not_blank(cls, v: str) -> str:
        if not v.strip():
            msg = "scope_filter must not be empty; it bounds the entire corpus"
            raise ValueError(msg)
        return v.strip()

    @property
    def openrouter_configured(self) -> bool:
        """True when an API key is present."""
        return self.openrouter_api_key is not None

    @property
    def chat_completions_url(self) -> str:
        """Full URL for OpenRouter chat completions."""
        return f"{str(self.openrouter_base_url).rstrip('/')}/chat/completions"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
