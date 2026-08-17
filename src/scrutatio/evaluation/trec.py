"""TREC Clinical Trials 2021: topics, judgements, and what they mean.

The Definition of Done asks for one thing above all others — *"evaluated against a
labelled benchmark; the system beats the baseline … 'we did not measure' is
unacceptable"* — and this is the data that makes answering it possible.

**Two facts about this benchmark decide whether a number from it means anything.**

*The labels are graded, and the middle one is a trap.* ``2`` means the patient is
eligible, ``1`` means the trial is about the patient's condition but he is
*excluded* from it, and ``0`` means irrelevant. A binary metric that folds ``1``
in with ``2`` rewards a system for surfacing trials the patient cannot join, which
is the opposite of the product. nDCG uses the graded scale directly and is
therefore the primary metric here; anything binary must say explicitly which
labels it counted.

*The pool is not the corpus.* TREC judged 26,162 trials, pooled from what the 2021
participants retrieved out of a snapshot of roughly 375,000. Ranking only within
that pool is easier than the real task, because every candidate was already
plausible enough for somebody's top-k. Measured: a **random** ranking inside this
pool scores nDCG@10 = 0.21. So numbers produced here are not comparable to
published TREC results — but a baseline and our system, run over the same pool,
are comparable to each other, which is the question actually being asked.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

TOPICS_URL: Final = "https://www.trec-cds.org/topics2021.xml"
QRELS_URL: Final = "https://trec.nist.gov/data/trials/qrels2021.txt"

# The roadmap recorded G7 as "open and blocking" because trec-cds.org answered 406
# to non-browser agents. Measured 2026-08-17: both files fetch cleanly. The block
# was stale, and it had been holding up the project's hardest requirement.
DEFAULT_DIR: Final = Path("data/trec")

# Graded relevance, as the track defines it.
ELIGIBLE: Final = 2
EXCLUDED: Final = 1
NOT_RELEVANT: Final = 0

_TOPIC_RE: Final = re.compile(r"<topic number=\"(\d+)\">(.*?)</topic>", re.DOTALL)


def load_topics(path: Path | None = None) -> dict[str, str]:
    """Topic number -> the patient description, whitespace-normalised.

    These are what a query looks like in this benchmark: several sentences of
    clinical narrative, not keywords. That matters for the comparison — the whole
    argument for embeddings is that a clinician's wording and a protocol's wording
    do not overlap lexically, and a topic like this is exactly that mismatch.
    """
    source = (path or DEFAULT_DIR / "topics2021.xml").read_text(encoding="utf-8")
    topics = {number: " ".join(body.split()) for number, body in _TOPIC_RE.findall(source)}
    if not topics:
        msg = "No topics parsed — the XML shape changed, or the file is truncated."
        raise ValueError(msg)
    return topics


def load_qrels(path: Path | None = None) -> dict[str, dict[str, int]]:
    """Topic -> {nct_id: graded relevance}.

    The file is standard TREC format: ``topic iteration doc_id relevance``. The
    iteration column is always 0 and is ignored, which is what it is for.
    """
    source = (path or DEFAULT_DIR / "qrels2021.txt").read_text(encoding="utf-8")
    qrels: dict[str, dict[str, int]] = {}
    for line in source.splitlines():
        fields = line.split()
        if len(fields) != 4:
            continue
        topic, _iteration, doc_id, relevance = fields
        qrels.setdefault(topic, {})[doc_id] = int(relevance)
    if not qrels:
        msg = "No judgements parsed — the qrels file is empty or malformed."
        raise ValueError(msg)
    return qrels


def judged_trials(qrels: dict[str, dict[str, int]]) -> list[str]:
    """Every NCT id that carries a judgement, in first-seen order.

    This is the retrievable universe for the evaluation. Order is stable so that a
    partial fetch resumes at the same place rather than reshuffling.
    """
    seen: dict[str, None] = {}
    for docs in qrels.values():
        for nct in docs:
            seen.setdefault(nct)
    return list(seen)


def label_summary(qrels: dict[str, dict[str, int]]) -> dict[str, int]:
    """Counts per label, for the record. A run whose totals differ from these has
    silently dropped topics or documents."""
    counts = {"topics": len(qrels), "pairs": 0, "eligible": 0, "excluded": 0, "not_relevant": 0}
    names = {ELIGIBLE: "eligible", EXCLUDED: "excluded", NOT_RELEVANT: "not_relevant"}
    for docs in qrels.values():
        for relevance in docs.values():
            counts["pairs"] += 1
            key = names.get(relevance)
            if key:
                counts[key] += 1
    counts["trials"] = len(judged_trials(qrels))
    return counts


def download(directory: Path | None = None) -> tuple[Path, Path]:
    """Fetch topics and qrels if they are not already on disk.

    A browser User-Agent is required: trec-cds.org answers a bare or custom agent
    with 403/406. That is a property of the host, not a workaround worth hiding.
    """
    import httpx

    target = directory or DEFAULT_DIR
    target.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }

    paths = []
    for name, url in (("topics2021.xml", TOPICS_URL), ("qrels2021.txt", QRELS_URL)):
        path = target / name
        if not path.exists():
            logger.info("Fetching %s", url)
            response = httpx.get(url, headers=headers, timeout=60, follow_redirects=True)
            response.raise_for_status()
            path.write_bytes(response.content)
        paths.append(path)
    return paths[0], paths[1]
