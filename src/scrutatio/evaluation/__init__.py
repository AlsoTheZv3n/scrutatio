"""Evaluation against TREC Clinical Trials 2021.

This package exists to answer the one question the rest of the project cannot
answer about itself: is retrieval over structured, embedded criteria actually
better than searching the prose with keywords? Everything else here is measured —
throughput, cost, taxonomy quality, latency — while the claim the product rests on
has never been tested.

Ordered so the cheap half comes first. The BM25 baseline needs only the raw text
that ``corpus`` lands, so the number to beat is available before a single token is
spent on extraction.
"""

from scrutatio.evaluation.trec import (
    ELIGIBLE,
    EXCLUDED,
    NOT_RELEVANT,
    judged_trials,
    label_summary,
    load_qrels,
    load_topics,
)

__all__ = [
    "ELIGIBLE",
    "EXCLUDED",
    "NOT_RELEVANT",
    "judged_trials",
    "label_summary",
    "load_qrels",
    "load_topics",
]
