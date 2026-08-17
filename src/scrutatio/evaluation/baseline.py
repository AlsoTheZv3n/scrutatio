"""The number to beat: BM25 over the trial text.

This is the honest competitor. ClinicalTrials.gov's own search is free, instant,
and already in every clinician's browser — so the question is not whether our
system produces plausible results, it is whether it produces *better* ones than
keyword matching over the same prose.

**The baseline is deliberately given every advantage.**

Two variants run. ``eligibility`` indexes only the criteria text, which is the
same information our system embeds — apples to apples. ``full`` adds the title,
the condition list and the brief summary, which is what a real keyword search
would actually have. The second is stronger and is the one that matters: beating
a baseline that was handicapped proves nothing, and the temptation to build one is
exactly why this file says so out loud.

Stopwords are removed and the topic is tokenised the same way as the documents.
Both help BM25. That is the point.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import duckdb

logger = logging.getLogger(__name__)

_TOKEN_RE: Final = re.compile(r"[a-z0-9]+")

# A short clinical-text stopword list. Kept small on purpose: aggressive removal
# would strip terms like "no" and "not" that carry eligibility meaning, and the
# baseline should be strong, not lobotomised.
_STOPWORDS: Final = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "s",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        # The last five are ubiquitous in this corpus, so they separate nothing.
        "patient",
        "patients",
        "subject",
        "subjects",
        "study",
    }
)


def tokenise(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, minus a small stopword list."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


_PATHS: Final = {
    "title": "$.protocolSection.identificationModule.briefTitle",
    "summary": "$.protocolSection.descriptionModule.briefSummary",
    "eligibility": "$.protocolSection.eligibilityModule.eligibilityCriteria",
}


def load_documents(
    db: duckdb.DuckDBPyConnection, *, variant: str = "full"
) -> tuple[list[str], list[str]]:
    """Return ``(nct_ids, texts)`` for every trial in this database.

    ``variant='eligibility'`` matches what our system indexes; ``'full'`` gives
    BM25 the title, conditions and summary as well.
    """
    if variant not in {"full", "eligibility"}:
        msg = f"variant must be 'full' or 'eligibility', not {variant!r}"
        raise ValueError(msg)

    select = [
        "nct_id",
        f"json_extract_string(raw, '{_PATHS['eligibility']}')",
    ]
    if variant == "full":
        select += [
            f"json_extract_string(raw, '{_PATHS['title']}')",
            f"json_extract_string(raw, '{_PATHS['summary']}')",
            "json_extract(raw, '$.protocolSection.conditionsModule.conditions')",
        ]

    rows = db.execute(f"SELECT {', '.join(select)} FROM bronze_studies").fetchall()  # noqa: S608

    ids: list[str] = []
    texts: list[str] = []
    for row in rows:
        parts = [row[1] or ""]
        if variant == "full":
            parts.append(row[2] or "")
            parts.append(row[3] or "")
            if row[4]:
                with contextlib.suppress(ValueError, TypeError):  # malformed payload
                    parts.append(" ".join(json.loads(row[4])))
        ids.append(row[0])
        texts.append(" ".join(p for p in parts if p))
    return ids, texts


def run_bm25(
    db: duckdb.DuckDBPyConnection,
    topics: Mapping[str, str],
    *,
    variant: str = "full",
    k: int = 1000,
) -> dict[str, dict[str, float]]:
    """Score every topic against every trial. Returns a TREC-style run.

    ``k=1000`` because recall@1000 is part of what the track reports, and cutting
    the run shorter would understate the baseline rather than measure it.
    """
    from rank_bm25 import BM25Okapi

    ids, texts = load_documents(db, variant=variant)
    logger.info("BM25 (%s): indexing %d documents", variant, len(ids))
    index = BM25Okapi([tokenise(t) for t in texts])

    run: dict[str, dict[str, float]] = {}
    for topic, description in topics.items():
        scores = index.get_scores(tokenise(description))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        # Zero-scoring documents are dropped: retrieving a document the query has
        # nothing in common with is noise, and trec_eval treats an absent document
        # exactly as it treats one ranked last.
        run[topic] = {ids[i]: float(scores[i]) for i in top if scores[i] > 0}
    return run


def restrict_to(
    run: Mapping[str, Mapping[str, float]], allowed: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Keep only documents in ``allowed``.

    Used to hold both systems to the same candidate set when one of them can only
    reach part of the pool — a trial that failed extraction has no vectors, and
    letting BM25 retrieve it while we cannot would measure the gap in coverage
    rather than the gap in ranking.
    """
    keep = set(allowed)
    return {t: {d: s for d, s in docs.items() if d in keep} for t, docs in run.items()}
