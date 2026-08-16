"""LLM extraction of structured eligibility criteria from trial prose."""

from scrutatio.extraction.client import (
    EligibilityExtractor,
    ExtractionError,
    TokenUsage,
    TruncatedResponseError,
    token_budget,
)
from scrutatio.extraction.runner import (
    DEFAULT_WORKERS,
    ExtractionOutcome,
    RunStats,
    extract_many,
)
from scrutatio.extraction.schema import (
    Criterion,
    CriterionKind,
    ExtractedEligibility,
    SchemaTooLargeError,
    flatten_schema,
    response_format,
)

__all__ = [
    "DEFAULT_WORKERS",
    "Criterion",
    "CriterionKind",
    "EligibilityExtractor",
    "ExtractedEligibility",
    "ExtractionError",
    "ExtractionOutcome",
    "RunStats",
    "SchemaTooLargeError",
    "TokenUsage",
    "TruncatedResponseError",
    "extract_many",
    "flatten_schema",
    "response_format",
    "token_budget",
]
