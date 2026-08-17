"""Live OpenRouter contract.

Every assertion here guards something a mock structurally cannot: that the model
still exists, that whichever provider serves the request still accepts our
flattened strict-mode schema, that ``reasoning: {enabled: false}`` is still
honoured, and that the response shape still survives our parser.

Cost is roughly two calls per run — fractions of a cent — and the calls are kept
small on purpose. Assertions are written to tolerate model nondeterminism: they
check invariants the prompt explicitly demands, never exact counts or wording.
"""

from __future__ import annotations

import pytest

from scrutatio.config import get_settings
from scrutatio.extraction.client import EligibilityExtractor
from scrutatio.extraction.schema import CriterionKind
from scrutatio.matching.judge import CriterionJudge

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not get_settings().openrouter_configured,
        reason="OPENROUTER_API_KEY is not set",
    ),
]

# Seven criteria spanning seven different kinds, so a collapse of the taxonomy is
# visible. Short, because every character is billed twice — once in, once in the
# restated answer.
_ELIGIBILITY = """Inclusion Criteria:
- Histologically confirmed stage IV non-small cell lung cancer
- ECOG performance status 0 or 1
- Absolute neutrophil count >= 1.5 x 10^9/L
- Measurable disease per RECIST v1.1

Exclusion Criteria:
- Untreated central nervous system metastases
- Prior treatment with an EGFR tyrosine kinase inhibitor
- Known active hepatitis B or C infection
"""

_VALID_KINDS = set(CriterionKind.__args__)  # type: ignore[attr-defined]

# Synthetic. The project forbids real patient data, and a live call leaves the
# machine.
_VIGNETTE = (
    "Synthetic case, not a real patient. 62-year-old with histologically confirmed "
    "stage IV non-small cell lung cancer, EGFR exon 19 deletion. ECOG 1. "
    "Brain MRI shows no metastases."
)

_TRIAL_CRITERIA: list[tuple[int, str, str, bool]] = [
    (0, "Histologically confirmed stage IV non-small cell lung cancer", "condition", False),
    (1, "ECOG performance status 0 or 1", "ecog", False),
    (2, "Absolute neutrophil count >= 1.5 x 10^9/L", "lab", False),
    (3, "Untreated central nervous system metastases", "comorbidity", True),
    (4, "Written informed consent obtained prior to any study procedure", "consent", False),
]


@pytest.fixture(scope="module")
def extracted() -> tuple[object, object]:
    """One real extraction call, shared by every assertion about it.

    Module-scoped deliberately: the assertions below are independent questions
    about the same response, and paying for one call per question would be waste.
    """
    with EligibilityExtractor() as extractor:
        result = extractor.extract(_ELIGIBILITY)
        return result, extractor.usage


class TestExtractionContract:
    def test_the_model_answers_and_the_schema_is_accepted(self, extracted: tuple) -> None:
        # A delisted model, a rejected schema or a provider that does not support
        # strict mode all land here as a raised error rather than a bad assertion.
        result, _ = extracted
        assert len(result.criteria) >= 5, f"expected most of 7 criteria, got {len(result.criteria)}"

    def test_every_kind_is_one_we_declared(self, extracted: tuple) -> None:
        result, _ = extracted
        kinds = {c.kind for c in result.criteria}
        assert kinds <= _VALID_KINDS, f"model invented kinds: {kinds - _VALID_KINDS}"

    def test_the_taxonomy_does_not_collapse_into_other(self, extracted: tuple) -> None:
        # The measured corpus sits at 4.06% `other`. It was 35% when the prompt
        # described only 12 of the 20 categories, and a criterion in `other` cannot
        # be matched against a patient later — so this is a recall guard, not style.
        result, _ = extracted
        kinds = [c.kind for c in result.criteria]
        assert len(set(kinds)) >= 4, f"only {len(set(kinds))} distinct kinds: {set(kinds)}"
        assert kinds.count("other") <= 1, f"{kinds.count('other')} criteria fell into `other`"

    def test_both_sections_are_distinguished(self, extracted: tuple) -> None:
        result, _ = extracted
        flags = {c.is_exclusion for c in result.criteria}
        assert flags == {True, False}, "exclusion criteria were not separated from inclusion"

    def test_thresholds_survive_verbatim(self, extracted: tuple) -> None:
        # The prompt says "preserve thresholds, units and time windows verbatim".
        # A paraphrased lab bound is unusable for matching.
        result, _ = extracted
        assert any("1.5" in c.text for c in result.criteria), "the ANC threshold was not preserved"

    def test_reasoning_is_actually_off(self, extracted: tuple) -> None:
        # The invariant that cost a full run on the previous platform. Measured at
        # ~1.0 with reasoning disabled and 1.54 over 30 trials with it on, with
        # individual samples above 3. The bound is loose because this is one call.
        _, usage = extracted
        assert usage.calls >= 1
        assert usage.reasoning_ratio < 2.0, (
            f"completion-to-answer ratio {usage.reasoning_ratio:.2f} — "
            "reasoning tokens are being billed again"
        )

    def test_nothing_was_truncated(self, extracted: tuple) -> None:
        # A truncated attempt burns its whole budget and returns nothing usable.
        _, usage = extracted
        assert usage.wasted_calls == 0


@pytest.fixture(scope="module")
def verdicts() -> list:
    """One real judge call, shared across the assertions about it.

    Module-scoped and defined at module level: pytest deprecated class-scoped
    fixtures written as instance methods, and this suite turns warnings into errors.
    """
    with CriterionJudge() as judge:
        return judge.judge(_VIGNETTE, _TRIAL_CRITERIA)


class TestJudgeContract:
    """One real judge call. The verdict shape is what the UI renders."""

    def test_every_criterion_gets_exactly_one_verdict(self, verdicts: list) -> None:
        # A missing row would read as "not applicable" in the UI when it actually
        # means "not answered".
        assert [v.ordinal for v in verdicts] == [c[0] for c in _TRIAL_CRITERIA]

    def test_verdicts_are_within_the_enum(self, verdicts: list) -> None:
        assert {v.verdict for v in verdicts} <= {"met", "not_met", "unclear"}

    def test_every_verdict_carries_a_rationale(self, verdicts: list) -> None:
        # The rationale is the whole product: a ranking a clinician cannot check is
        # not decision support.
        assert all(v.rationale.strip() for v in verdicts)

    def test_a_stated_fact_is_decided_not_hedged(self, verdicts: list) -> None:
        # The vignette says ECOG 1; the criterion allows 0 or 1. If the model
        # cannot resolve that, `unclear` has stopped meaning "cannot tell".
        ecog = next(v for v in verdicts if v.ordinal == 1)
        assert ecog.verdict == "met", f"ECOG 1 vs '0 or 1' came back {ecog.verdict}"

    def test_an_administrative_criterion_stays_unclear(self, verdicts: list) -> None:
        # Measured over 88 criteria of three real trials: consent, compliance and
        # contraception came back 100% unclear, because a patient description
        # structurally cannot address them. That is the correct answer, and the
        # prompt asks for it — a `met` here would be the model guessing.
        consent = next(v for v in verdicts if v.ordinal == 4)
        assert consent.verdict == "unclear", f"consent was judged {consent.verdict}"
