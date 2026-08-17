"""Judging a patient description against a trial's criteria.

One call per trial covering all of its criteria, for the same reason extraction
works that way: the per-criterion fan-out costs close to an order of magnitude
more tokens and judges each criterion without sight of the others.

The prompt is built around a finding from real data, not a preference. Measured
over 88 criteria of three real trials, 73% of verdicts came back ``unclear`` —
and the split is not random. Categories a patient description structurally
cannot address (consent, compliance, contraception) came back 100% unclear,
while ECOG, age, stage and biomarkers came back 0%. So ``unclear`` is not the
model hedging; for a large part of any trial's criteria it is the only honest
answer, and the prompt asks for it explicitly rather than pushing the model
toward a guess.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Final, Literal

from pydantic import BaseModel, Field, ValidationError

from scrutatio.config import Settings, get_settings
from scrutatio.extraction.schema import flatten_schema
from scrutatio.matching.schema import CriterionVerdict
from scrutatio.openrouter import OpenRouterClient, OpenRouterError, message_text

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

JUDGE_PROMPT: Final = (
    "You judge whether a patient meets each eligibility criterion of a clinical trial.\n"
    "\n"
    "For every criterion you are given, return exactly one verdict:\n"
    "- met: the description satisfies it. For an EXCLUSION criterion, 'met' means the "
    "patient is NOT excluded by it.\n"
    "- not_met: the description contradicts it. For an EXCLUSION criterion, 'not_met' "
    "means the patient IS excluded.\n"
    "- unclear: the description does not contain what is needed to decide.\n"
    "\n"
    "Prefer 'unclear' over guessing. A missing lab value, an unstated duration, an absent "
    "staging detail or an unmentioned prior therapy is unclear — not met, and not not_met. "
    "Many criteria are administrative (consent, protocol adherence, contraception) and a "
    "patient description will never address them; those are unclear too, and that is the "
    "correct answer rather than a failure.\n"
    "\n"
    "Give one short rationale per criterion, referring to what the description does or does "
    "not say. Copy each ordinal exactly as given. Return a verdict for every criterion."
)

# The judge restates nothing: per criterion it emits an ordinal, a verdict and a
# sentence. Measured on real trials at roughly 40 output tokens per criterion,
# with a floor for trials that have very few.
_TOKENS_PER_CRITERION: Final = 70
_MIN_TOKENS: Final = 1500
_MAX_TOKENS: Final = 32000


class _Verdict(BaseModel):
    """The per-criterion shape the model must fill. Flat, no optionals."""

    ordinal: int = Field(description="The criterion's ordinal, copied from the input.")
    verdict: Literal["met", "not_met", "unclear"]
    rationale: str = Field(description="One sentence referring to the patient description.")


class _Judged(BaseModel):
    verdicts: list[_Verdict] = Field(description="One entry per criterion given.")


def _response_format() -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {"name": "verdicts", "strict": True, "schema": flatten_schema(_Judged)},
    }


def judge_prompt_version() -> str:
    """Short hash of everything that changes a verdict.

    The judgement cache is keyed on this, so editing the prompt or the schema
    invalidates exactly the cached verdicts it affects — the same mechanism the
    extraction signature provides for Silver, for the same reason.
    """
    import hashlib

    cfg = get_settings()
    material = json.dumps(
        {"model": cfg.matching_model, "prompt": JUDGE_PROMPT, "schema": flatten_schema(_Judged)},
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


class CriterionJudge:
    """Turns a patient description plus a trial's criteria into verdicts."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: OpenRouterClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._owns_client = client is None
        self._client = client or OpenRouterClient(self._settings)
        self.model_name = self._settings.matching_model

    def __enter__(self) -> CriterionJudge:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def judge(
        self,
        patient_text: str,
        criteria: Sequence[tuple[int, str, str, bool]],
    ) -> list[CriterionVerdict]:
        """Judge one trial's criteria, given as ``(ordinal, text, kind, is_exclusion)``.

        Every criterion comes back with a verdict. A criterion the model omitted
        is returned as ``unclear`` with that stated as the reason, rather than
        dropped — a missing row in the UI would read as "not applicable" when it
        actually means "not answered".
        """
        if not criteria:
            return []

        listing = "\n".join(
            f"{ordinal}. [{'EXCLUSION' if is_exclusion else 'INCLUSION'}] {text}"
            for ordinal, text, _kind, is_exclusion in criteria
        )
        budget = max(_MIN_TOKENS, min(len(criteria) * _TOKENS_PER_CRITERION, _MAX_TOKENS))

        payload = self._client.build_payload(
            model=self._settings.matching_model,
            system=JUDGE_PROMPT,
            user=f"PATIENT:\n{patient_text}\n\nCRITERIA:\n{listing}",
            max_tokens=budget,
            response_format=_response_format(),
        )
        body = self._client.post(payload)

        try:
            text = message_text(body["choices"][0]["message"])
            judged = _Judged.model_validate_json(text)
        except (KeyError, IndexError) as exc:
            msg = f"Unexpected response envelope: {json.dumps(body)[:300]}"
            raise OpenRouterError(msg) from exc
        except ValidationError as exc:
            msg = f"Judge returned schema-invalid JSON despite strict mode: {exc}"
            raise OpenRouterError(msg) from exc

        by_ordinal = {v.ordinal: v for v in judged.verdicts}
        missing = [o for o, _, _, _ in criteria if o not in by_ordinal]
        if missing:
            logger.warning("Judge omitted %d of %d criteria", len(missing), len(criteria))

        return [
            CriterionVerdict(
                ordinal=ordinal,
                text=text,
                kind=kind,  # type: ignore[arg-type]
                is_exclusion=is_exclusion,
                verdict=by_ordinal[ordinal].verdict if ordinal in by_ordinal else "unclear",
                rationale=(
                    by_ordinal[ordinal].rationale
                    if ordinal in by_ordinal
                    else "The model returned no verdict for this criterion."
                ),
            )
            for ordinal, text, kind, is_exclusion in criteria
        ]
