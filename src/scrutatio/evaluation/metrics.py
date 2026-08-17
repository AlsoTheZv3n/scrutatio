"""Scoring a run, using the official trec_eval rather than our own arithmetic.

``pytrec_eval`` wraps the C implementation NIST publishes. That choice is not
convenience: the claim this whole exercise exists to test is "the system beats the
baseline", and a hand-rolled nDCG that is subtly wrong would make that claim
worthless in exactly the way that is hardest to notice.

The harness was validated against known answers before any real number was
produced — ranking by the true label scores 1.0000, ranking by its negation scores
0.0000. A metric that cannot reproduce those is not measuring what it claims.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

# nDCG first, because the labels are graded and it is the only common metric that
# uses that grading directly. P@10 and recall are reported alongside for shape,
# not as the verdict.
DEFAULT_MEASURES: Final = frozenset(
    {"ndcg_cut_10", "ndcg_cut_100", "recall_100", "recall_1000", "P_10"}
)

# Measured on this pool with a seeded shuffle: a ranking with no information
# scores this much, because every candidate was pooled from someone's top-k. Any
# result must be read against this floor, not against zero.
RANDOM_FLOOR_NDCG10: Final = 0.2135


def evaluate(
    qrels: Mapping[str, Mapping[str, int]],
    run: Mapping[str, Mapping[str, float]],
    *,
    measures: frozenset[str] = DEFAULT_MEASURES,
) -> dict[str, float]:
    """Mean of each measure across topics.

    A topic present in ``qrels`` but absent from ``run`` scores zero rather than
    being skipped — otherwise a system that answers only the easy topics would
    outscore one that attempts all of them.
    """
    import pytrec_eval

    evaluator = pytrec_eval.RelevanceEvaluator(
        {t: dict(d) for t, d in qrels.items()}, set(measures)
    )
    per_topic = evaluator.evaluate({t: dict(d) for t, d in run.items()})

    missing = set(qrels) - set(per_topic)
    if missing:
        logger.warning("%d topics produced no results and score 0", len(missing))

    totals = dict.fromkeys(measures, 0.0)
    for scores in per_topic.values():
        for measure in measures:
            totals[measure] += scores.get(measure, 0.0)

    denominator = len(qrels)  # not len(per_topic) — see the docstring
    return {measure: totals[measure] / denominator for measure in sorted(measures)}


def compare(
    baseline: Mapping[str, float], system: Mapping[str, float]
) -> dict[str, dict[str, float]]:
    """Side by side, with the delta. The delta is the whole point."""
    return {
        measure: {
            "baseline": baseline.get(measure, 0.0),
            "system": system.get(measure, 0.0),
            "delta": system.get(measure, 0.0) - baseline.get(measure, 0.0),
        }
        for measure in sorted(set(baseline) | set(system))
    }
