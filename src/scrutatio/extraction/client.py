"""LLM extraction over OpenRouter.

One call per trial covering all criteria — not one call per criterion. The
per-criterion fan-out costs close to an order of magnitude more tokens for worse
Micro-F1, so the cheap shape is also the accurate one.

The HTTP transport is **not** here. It lives in ``scrutatio.clients.openrouter``
and is shared with the matching judge. It used to be duplicated: this file carried
its own copy of the retry loop, the header block, the keep-alive-padding parser
and the ``Retry-After`` handling — six elements, roughly a hundred lines, and two
mutually unaware exception hierarchies. Nothing had gone wrong yet, but every
lesson learned against the endpoint had to be applied twice or it applied to one
caller only.

What remains here is what is genuinely about extraction:

1. **The token budget.** ``max_tokens`` bounds everything the model writes, and a
   truncation discards the whole response. The ratio below is measured, and the
   budget is an invitation rather than a free ceiling — see ``_OUTPUT_RATIO``.
2. **Truncation is not a schema violation.** ``finish_reason == "length"`` means
   strict mode worked and the answer was simply cut off, so it gets its own error
   type and a retry with a doubled budget.
3. **Reasoning tokens count against the output budget.** ``gpt-oss-120b`` cost a
   full run this way: thinking consumed the ceiling and a third of responses were
   cut off mid-JSON. ``usage`` is accumulated per extractor so a calibration run
   can compare completion tokens against the size of the JSON that actually came
   back. A large gap means the model is reasoning, and for structured extraction
   that is pure waste.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Final, Self

import httpx
from pydantic import ValidationError

from scrutatio.clients.openrouter import OpenRouterClient, OpenRouterError, message_text
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
    "Choose `kind` by what is being tested, not by which section it appears in. "
    "Every category is listed here; pick the closest one.\n"
    "\n"
    "What the patient has:\n"
    "- condition: the disease itself — its diagnosis, histology, subtype or primary site\n"
    "- stage: disease stage, grade, or extent such as metastatic, locally advanced, resectable\n"
    "- biomarker: a molecular, genetic or immunohistochemical marker, mutation or expression\n"
    "- comorbidity: another illness or condition alongside the cancer, including prior malignancy\n"
    "- infection: HIV, hepatitis, tuberculosis, active infection\n"
    "\n"
    "Measurements:\n"
    "- lab: a numeric laboratory threshold with a value and unit\n"
    "- organ_function: adequate renal, hepatic, cardiac or marrow function stated qualitatively\n"
    "- ecog: ECOG, WHO or Karnofsky performance status\n"
    "- age: a minimum or maximum age\n"
    "- life_expectancy: a minimum expected survival\n"
    "- measurable_disease: RECIST-measurable or evaluable lesions\n"
    "\n"
    "Treatment history and constraints:\n"
    "- prior_therapy: a treatment the patient has or has not received before, including lines of\n"
    "  therapy, prior drug classes, and prior radiotherapy or systemic treatment\n"
    "- concurrent_therapy: participation in another trial, or use of a drug during this one\n"
    "- procedure: surgery, biopsy, transplant, radiotherapy as an event the patient underwent\n"
    "- washout: a required interval since a prior treatment or event ('at least 28 days since')\n"
    "- allergy: hypersensitivity or intolerance to a substance\n"
    "\n"
    "Administrative:\n"
    "- reproductive: pregnancy, lactation, contraception, childbearing potential, fertility\n"
    "- consent: ability or willingness to give informed consent or assent\n"
    "- compliance: willingness to attend visits, adhere to the protocol, be reachable\n"
    "\n"
    "Use `other` only when no listed kind fits. It is the last resort, not the default. "
    "A criterion placed in `other` cannot be matched against a patient later, so reaching for "
    "it is a real loss — if a criterion is about the disease, a marker, a measurement or a past "
    "treatment, one of the categories above applies."
)

MIN_MAX_TOKENS: Final = 2000
MAX_TOKEN_CEILING: Final = 32000

# A flat allowance for tokens the model emits that are not the answer. With
# reasoning disabled the measured completion-to-answer ratio is ~1.0, so this is
# small; it exists because `max_tokens` bounds everything the model writes, not
# just the JSON, and a truncation discards the whole response.
#
# It was briefly 2,000, sized against a model that *was* reasoning. That is the
# wrong fix for the wrong problem — see the note on budget size below.
_REASONING_HEADROOM: Final = 500

# The output restates every criterion plus its metadata, so it runs longer than
# the input prose. Measured over 4,000 real trials: 26.6 criteria per trial at
# ~190 output characters each, against a mean input of 3,341 characters.
#
# The old value of 3.0 was tuned against Databricks, which reserved `max_tokens`
# from the per-minute allowance *before* admitting a request. It was also
# measurably too tight here: NCT06999707 returned 8,308 characters of JSON,
# needing ~2,518 output tokens against a budget of 2,495 — truncated by 23
# tokens, and paid for a full retry.
#
# **Over-reserving is not free**, which is the opposite of what an earlier
# version of this comment claimed. OpenRouter does not pre-reserve, so a large
# ceiling costs nothing directly — but the model expands to fill it. Raising the
# floor to 4,000 and the allowance to 2,000 nearly doubled output tokens per
# trial (1,743 to 3,355) across the same 30 trials, for 681 versus 676 criteria.
# The budget is not a free upper bound; it is an invitation.
#
# The real fix was upstream: disabling reasoning, which cut output 2.5-3.8x. 4.5
# then covers the worst observed answer expansion with room to spare.
_OUTPUT_RATIO: Final = 4.5
_CHARS_PER_TOKEN: Final = 4


class ExtractionError(OpenRouterError):
    """The model's answer could not be turned into typed criteria.

    Subclassing ``OpenRouterError`` is not cosmetic. The two were unrelated types
    raised by two copies of the same transport, so a handler catching one silently
    missed the other — and the API's error table maps ``OpenRouterError`` to 502.
    An extraction failure arriving there unmapped would have become a 500, telling
    the caller "our fault, do not retry" about an upstream problem.

    For anyone adding a handler: ``except ExtractionError`` does **not** catch a
    bare ``OpenRouterError`` from the transport — a subclass never catches its
    base. ``extraction/runner.py`` therefore catches ``OpenRouterError``, because
    its throttling detection reads "429" out of the message and a 429 is raised by
    the transport rather than here.
    """


class TruncatedResponseError(ExtractionError):
    """The model hit the token ceiling before closing the JSON.

    Distinct from a schema violation: strict mode worked, the answer was simply
    cut off. Reported separately because the fix is a bigger budget, not a
    different prompt.
    """


@dataclass
class TokenUsage:
    """Accumulated usage across one extractor's lifetime.

    ``completion_tokens`` far exceeding ``json_chars / 4`` is the signature of a
    reasoning model spending the output budget on thinking. That is the first
    thing the calibration gate checks.
    """

    calls: int = 0
    wasted_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    json_chars: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self, usage: dict[str, Any] | None, json_chars: int, *, wasted: bool = False
    ) -> None:
        with self._lock:
            self.calls += 1
            self.wasted_calls += int(wasted)
            self.json_chars += json_chars
            if usage:
                self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.completion_tokens += int(usage.get("completion_tokens") or 0)

    @property
    def reasoning_ratio(self) -> float:
        """Completion tokens per token of JSON actually returned.

        Near 1.0 means the model emitted only the answer. Substantially above
        that means tokens went somewhere the answer did not.
        """
        answer_tokens = self.json_chars / _CHARS_PER_TOKEN
        return self.completion_tokens / answer_tokens if answer_tokens else 0.0


def token_budget(eligibility_text: str) -> int:
    """Output-token ceiling: the answer scaled to the input, plus a flat
    allowance for whatever the model emits that is not the answer."""
    answer = len(eligibility_text) / _CHARS_PER_TOKEN * _OUTPUT_RATIO
    estimated = int(answer + _REASONING_HEADROOM)
    return max(MIN_MAX_TOKENS, min(estimated, MAX_TOKEN_CEILING))


class EligibilityExtractor:
    """Turns eligibility prose into validated, typed predicates."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._model = model or self._settings.extraction_model
        self._max_tokens = max_tokens
        # The shared transport owns the credential check, the retry policy and the
        # httpx client. An injected client is forwarded, so it stays the caller's
        # to close.
        self._client = OpenRouterClient(self._settings, client=client)
        self.usage = TokenUsage()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

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
        # Reasoning and provider routing are applied by the shared transport from
        # settings, so both callers get them without either having to remember.
        #
        # Routing is deliberately NOT part of the extraction signature. It is
        # infrastructure, not a model change — the model id is identical, and
        # signing it would re-queue every trial already extracted for the sake of a
        # latency preference. The honest caveat: the corpus is already
        # provider-heterogeneous, because default routing spread the first 7,679
        # trials across at least eight of them. Pinning makes what follows more
        # consistent, not less. Guarded by a test.
        payload = self._client.build_payload(
            model=self._model,
            system=SYSTEM_PROMPT,
            user=eligibility_text,
            max_tokens=max_tokens,
            response_format=response_format(ExtractedEligibility, "eligibility"),
            reasoning=self._settings.extraction_reasoning,
        )

        body = self._client.post(payload)

        try:
            choice = body["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError) as exc:
            msg = f"Unexpected response envelope: {json.dumps(body)[:300]}"
            raise ExtractionError(msg) from exc

        if choice.get("finish_reason") == "length":
            # Record before raising: a truncated attempt burns its whole budget
            # and returns nothing usable. Counting the tokens but not the answer
            # is what makes the waste visible in `reasoning_ratio` instead of
            # disappearing from the numbers entirely.
            self.usage.record(body.get("usage"), 0, wasted=True)
            msg = f"Response hit the {max_tokens}-token ceiling before closing the JSON"
            raise TruncatedResponseError(msg)

        text = message_text(message)
        self.usage.record(body.get("usage"), len(text))

        try:
            return ExtractedEligibility.model_validate_json(text)
        except ValidationError as exc:
            msg = f"Model returned schema-invalid JSON despite strict mode: {exc}"
            raise ExtractionError(msg) from exc
