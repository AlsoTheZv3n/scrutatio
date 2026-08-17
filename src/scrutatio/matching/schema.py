"""The result contract: what matching emits and what the UI reads.

Defined before the matching pipeline exists, deliberately. The frontend is being
built against this boundary right now, and if both sides invent a shape
independently the difference becomes a translation layer nobody wanted.

**The wire format is snake_case on both sides.** Python emits it natively and
TypeScript only has to declare it, so there is no alias configuration to get
wrong and no field that silently arrives as `undefined`.

The verdict per criterion is the product here, not the ranking. A score a
clinician cannot check is worse than no answer, because it invites trust it has
not earned — which is also why ``rationale`` is required rather than optional and
why ``unclear`` is a first-class verdict rather than a low-confidence ``met``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from scrutatio.extraction.schema import CriterionKind

# Three states, and the third is not a hedge. "unclear" means the criterion
# cannot be decided from the patient description given — a missing lab value, an
# unstated duration. Those are exactly the items a human should look at, so they
# are a useful answer rather than a failure to produce one.
Verdict = Literal["met", "not_met", "unclear"]

DISCLAIMER = (
    "Decision support, not medical advice. This is a prioritised list of candidate "
    "trials with per-criterion evidence for review by a qualified healthcare "
    "professional — not an eligibility determination. Eligibility must always be "
    "verified with the study team. Data comes from ClinicalTrials.gov and may be "
    "out of date. Do not enter real patient data."
)


class CriterionVerdict(BaseModel):
    """One eligibility criterion, judged against the patient description."""

    ordinal: int = Field(description="Stable key within the trial, from Silver.")
    text: str = Field(description="The criterion as a single self-contained sentence.")
    kind: CriterionKind
    is_exclusion: bool
    verdict: Verdict
    rationale: str = Field(
        description="Why this verdict, referring to the patient description. Never empty."
    )


class MatchedOn(BaseModel):
    """Why the trial was retrieved at all — the nearest criterion by embedding."""

    ordinal: int
    text: str
    score: float = Field(ge=-1.0, le=1.0, description="Cosine similarity.")


class VerdictCounts(BaseModel):
    """Rendered as a one-line summary; `unclear` is the work list."""

    met: int = 0
    not_met: int = 0
    unclear: int = 0


class TrialMatch(BaseModel):
    """One candidate trial with its full criterion breakdown."""

    nct_id: str
    title: str
    phase: str | None = None
    overall_status: str
    locations: list[str] = Field(default_factory=list)
    matched_on: MatchedOn
    counts: VerdictCounts
    criteria: list[CriterionVerdict] = Field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://clinicaltrials.gov/study/{self.nct_id}"


class MatchResponse(BaseModel):
    """The whole payload for one query. This is the JSON the UI consumes."""

    query: str
    generated_at: datetime
    model: str
    # Carried in the payload rather than hardcoded in the UI so it cannot be
    # dropped by a refactor on the frontend. The Definition of Done requires it
    # visible without scrolling.
    disclaimer: str = DISCLAIMER
    trials: list[TrialMatch] = Field(default_factory=list)
