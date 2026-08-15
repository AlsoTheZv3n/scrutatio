"""LLM extraction against a Databricks Foundation Model serving endpoint.

One call per trial covering all criteria — not one call per criterion. The
per-criterion fan-out costs close to an order of magnitude more tokens for
worse Micro-F1, and the rate limits on Free Edition are undocumented, so the
cheap shape is also the safe one.

The response shape is not uniform across models. Reasoning models (the
``gpt-oss`` family) return ``message.content`` as a list of typed parts —
``{"type": "reasoning", ...}`` followed by ``{"type": "text", ...}`` — while the
others return a plain string. A client that assumes ``str`` breaks on half the
endpoints, so both are handled.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Final

import httpx
from pydantic import ValidationError

from scrutatio.config import Settings, get_settings
from scrutatio.extraction.schema import ExtractedEligibility, response_format

logger = logging.getLogger(__name__)

SYSTEM_PROMPT: Final = (
    "You extract eligibility criteria from clinical trial protocol text.\n"
    "\n"
    "Split compound sentences so that each criterion expresses exactly one testable "
    "predicate: a condition and a biomarker mentioned together become two criteria. "
    "Preserve thresholds, units and time windows verbatim. Mark every criterion from "
    "the exclusion section with is_exclusion=true. Do not invent criteria that are "
    "not in the text.\n"
    "\n"
    "Choose `kind` by what is being tested, not by which section it appears in:\n"
    "- reproductive: pregnancy, lactation, contraception, childbearing potential, fertility\n"
    "- consent: ability or willingness to give informed consent or assent\n"
    "- compliance: willingness to attend visits, adhere to the protocol, be reachable\n"
    "- procedure: surgery, biopsy, transplant, radiotherapy as an event the patient underwent\n"
    "- washout: a required interval since a prior treatment or event ('at least 28 days since')\n"
    "- concurrent_therapy: participation in another trial, or use of a drug during this one\n"
    "- organ_function: adequate renal, hepatic, cardiac or marrow function stated qualitatively\n"
    "- lab: a numeric laboratory threshold with a value and unit\n"
    "- infection: HIV, hepatitis, tuberculosis, active infection\n"
    "- measurable_disease: RECIST-measurable or evaluable lesions\n"
    "- life_expectancy: a minimum expected survival\n"
    "- allergy: hypersensitivity or intolerance to a substance\n"
    "\n"
    "Use `other` only when no listed kind fits. It is the last resort, not the default."
)

# Output tokens are pre-reserved from the per-minute budget, so an oversized
# ceiling throttles throughput even when unused — but too tight a ceiling
# truncates the JSON mid-field on criteria-heavy trials. The budget is therefore
# scaled to the input and retried once on truncation.
MIN_MAX_TOKENS: Final = 1000
MAX_TOKEN_CEILING: Final = 8000

# The output restates every criterion plus its metadata, so it runs longer than
# the input prose. Measured on a real trial: 1,788 output tokens for 1,629 input,
# a ratio of 1.86. The budget is not just a safety net — the endpoint reserves it
# against the per-minute output allowance before admitting the request, so padding
# it costs throughput directly. 2.0 leaves a margin; TruncatedResponseError doubles
# it for the rare trial that needs more.
_OUTPUT_RATIO: Final = 2.0
_CHARS_PER_TOKEN: Final = 4

_RETRY_STATUS: Final = frozenset({429, 500, 502, 503, 504})


class ExtractionError(RuntimeError):
    """The endpoint failed, or returned something that is not valid output."""


class TruncatedResponseError(ExtractionError):
    """The model hit the token ceiling before closing the JSON.

    Distinct from a schema violation: strict mode worked, the answer was simply
    cut off. Reported separately because the fix is a bigger budget, not a
    different prompt.
    """


def token_budget(eligibility_text: str) -> int:
    """Output-token ceiling scaled to the length of the input."""
    estimated = int(len(eligibility_text) / _CHARS_PER_TOKEN * _OUTPUT_RATIO)
    return max(MIN_MAX_TOKENS, min(estimated, MAX_TOKEN_CEILING))


def _message_text(message: dict[str, Any]) -> str:
    """Pull the assistant text out of either response shape."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Reasoning parts are deliberately discarded; only `text` is the answer.
        parts = [
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        ]
        if parts:
            return "".join(parts)
    msg = f"Could not find assistant text in message of type {type(content).__name__}"
    raise ExtractionError(msg)


class EligibilityExtractor:
    """Turns eligibility prose into validated, typed predicates."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        if not self._settings.databricks_configured:
            msg = "DATABRICKS_HOST and DATABRICKS_TOKEN must be set to run extraction"
            raise ValueError(msg)
        self._max_tokens = max_tokens
        self._owns_client = client is None
        # Measured on real trials: 25.6s for a single call on a criteria-heavy
        # study. The 30s ingest timeout left ~4s of headroom, which is not a
        # margin across 11,200 trials.
        self._client = client or httpx.Client(timeout=self._settings.llm_timeout_seconds)

    def __enter__(self) -> EligibilityExtractor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @property
    def _headers(self) -> dict[str, str]:
        token = self._settings.databricks_token
        assert token is not None  # guaranteed by databricks_configured
        return {
            "Authorization": f"Bearer {token.get_secret_value()}",
            "Content-Type": "application/json",
        }

    def extract(self, eligibility_text: str) -> ExtractedEligibility:
        """Extract structured criteria from one trial's eligibility section.

        Retries once with a doubled token budget if the first attempt is cut
        off — criteria-heavy trials are common enough that failing outright
        would lose real data.
        """
        if not eligibility_text.strip():
            return ExtractedEligibility(criteria=[])

        budget = self._max_tokens or token_budget(eligibility_text)

        try:
            return self._attempt(eligibility_text, budget)
        except TruncatedResponseError:
            retry_budget = min(budget * 2, MAX_TOKEN_CEILING)
            if retry_budget <= budget:
                raise
            logger.warning(
                "Extraction truncated at %d tokens, retrying with %d", budget, retry_budget
            )
            return self._attempt(eligibility_text, retry_budget)

    def _attempt(self, eligibility_text: str, max_tokens: int) -> ExtractedEligibility:
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": eligibility_text},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": response_format(ExtractedEligibility, "eligibility"),
        }

        body = self._post(payload)

        try:
            choice = body["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError) as exc:
            msg = f"Unexpected response envelope: {json.dumps(body)[:300]}"
            raise ExtractionError(msg) from exc

        if choice.get("finish_reason") == "length":
            msg = f"Response hit the {max_tokens}-token ceiling before closing the JSON"
            raise TruncatedResponseError(msg)

        text = _message_text(message)

        try:
            return ExtractedEligibility.model_validate_json(text)
        except ValidationError as exc:
            msg = f"Model returned schema-invalid JSON despite strict mode: {exc}"
            raise ExtractionError(msg) from exc

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = self._settings.serving_endpoint_url(self._settings.chat_endpoint)
        last: str | None = None

        for attempt in range(self._settings.max_retries + 1):
            response: httpx.Response | None = None
            try:
                response = self._client.post(url, json=payload, headers=self._headers)
            except httpx.HTTPError as exc:
                last = str(exc)
            else:
                if response.status_code == httpx.codes.OK:
                    return response.json()
                if response.status_code not in _RETRY_STATUS:
                    msg = f"Serving endpoint returned {response.status_code}: {response.text[:300]}"
                    raise ExtractionError(msg)
                last = f"HTTP {response.status_code}"

            if attempt < self._settings.max_retries:
                delay = self._retry_delay(response, attempt)
                logger.warning("Extraction retry %d in %.0fs: %s", attempt + 1, delay, last)
                time.sleep(delay)

        msg = f"Extraction failed after {self._settings.max_retries + 1} attempts: {last}"
        raise ExtractionError(msg)

    @staticmethod
    def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
        """Honour ``Retry-After`` when the endpoint sends it; else back off."""
        if response is not None:
            header = response.headers.get("retry-after")
            if header:
                try:
                    return float(header)
                except ValueError:
                    pass
        return 2.0**attempt
