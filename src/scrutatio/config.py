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

# Embeddings run locally — no API, no per-vector cost. gte has an 8192-token
# window against a measured mean of ~835 tokens per eligibility text, so nothing
# is truncated. BGE's 512-token window would have forced chunking.
DEFAULT_EMBEDDING_MODEL = "Alibaba-NLP/gte-large-en-v1.5"
EMBEDDING_DIMENSIONS = 1024

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
