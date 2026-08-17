"""Retrieve then rerank: patient text in, ranked trials with verdicts out.

Two stages, because they answer different questions and cost different amounts.

**Retrieval** is a cosine query over 351,745 criterion vectors — milliseconds,
free, and it finds trials through a *single* criterion. That is enough to
shortlist and nothing more: a trial can be retrieved on a criterion the patient
matches perfectly while being excluded by another the retrieval never looked at.

**Reranking** therefore judges every criterion of every shortlisted trial. That
is one model call per trial, and it is where both the latency and the cost live —
which is why K is small and why verdicts are cached per trial rather than per
query.

The candidates are judged concurrently. The measured per-call latency is around
ten seconds, so at K=10 serial execution would spend a minute and a half against
a five-second target.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from scrutatio.clients.openrouter import OpenRouterError
from scrutatio.matching.judge import CriterionJudge, judge_prompt_version
from scrutatio.matching.schema import (
    CriterionVerdict,
    MatchedOn,
    MatchResponse,
    TrialMatch,
    VerdictCounts,
)
from scrutatio.storage.bronze import trial_metadata
from scrutatio.storage.cache import cached_verdicts, ensure_cache, query_hash, store_verdicts
from scrutatio.storage.gold import search_criteria
from scrutatio.storage.silver import extraction_signature, trial_criteria

if TYPE_CHECKING:
    import duckdb

    from scrutatio.embeddings import CriterionEncoder

logger = logging.getLogger(__name__)

# K after rerank. The ROADMAP's latency target is under five seconds warm, and
# every candidate costs a model call, so this is a budget rather than a taste.
DEFAULT_K = 10

# Retrieval pulls more than K because a trial can surface several times through
# different criteria; `search_criteria` already collapses to one row per trial,
# so this is headroom for trials that turn out to have no criteria in Silver.
_RETRIEVE_MULTIPLIER = 2


@dataclass(frozen=True)
class MatchTiming:
    """Where the wall clock went. Reported because the target is a number."""

    embed_seconds: float
    retrieve_seconds: float
    judge_seconds: float
    cache_hits: int
    judged: int


def run_match(
    *,
    db: duckdb.DuckDBPyConnection,
    encoder: CriterionEncoder,
    judge: CriterionJudge,
    text: str,
    k: int = DEFAULT_K,
    workers: int = DEFAULT_K,
) -> tuple[MatchResponse, MatchTiming]:
    """Rank trials for a patient description and judge each candidate's criteria."""
    ensure_cache(db)
    signature = extraction_signature()
    prompt_version = judge_prompt_version()
    key = query_hash(text)

    t0 = time.monotonic()
    query_vector = encoder.encode_query(text)
    t1 = time.monotonic()

    hits = search_criteria(
        db,
        query_vector,
        signature=signature,
        model=encoder.model_name,
        k=k * _RETRIEVE_MULTIPLIER,
    )[:k]
    nct_ids = [h[0] for h in hits]
    meta = trial_metadata(db, nct_ids)
    criteria = trial_criteria(db, nct_ids, signature)
    t2 = time.monotonic()

    cached = cached_verdicts(db, key, nct_ids, signature=signature, prompt_version=prompt_version)
    to_judge = [n for n in nct_ids if n not in cached]

    def judge_one(nct_id: str) -> tuple[str, list[CriterionVerdict]]:
        try:
            return nct_id, judge.judge(text, criteria.get(nct_id, []))
        except OpenRouterError as exc:
            # One candidate failing must not lose the other nine. The trial still
            # appears, with every criterion marked unanswered and the reason
            # visible, rather than silently dropping out of the ranking.
            logger.warning("Judging %s failed: %s", nct_id, str(exc)[:160])
            return nct_id, [
                CriterionVerdict(
                    ordinal=ordinal,
                    text=ctext,
                    kind=kind,  # type: ignore[arg-type]
                    is_exclusion=is_exclusion,
                    verdict="unclear",
                    rationale="Judging this trial failed; no verdict was produced.",
                )
                for ordinal, ctext, kind, is_exclusion in criteria.get(nct_id, [])
            ]

    if to_judge:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(to_judge)))) as pool:
            for nct_id, verdicts in pool.map(judge_one, to_judge):
                cached[nct_id] = verdicts
                if verdicts:
                    store_verdicts(
                        db,
                        key,
                        nct_id,
                        verdicts,
                        signature=signature,
                        prompt_version=prompt_version,
                    )
    t3 = time.monotonic()

    trials: list[TrialMatch] = []
    for nct_id, score, ordinal, matched_text in hits:
        info = meta.get(nct_id)
        verdicts = cached.get(nct_id, [])
        trials.append(
            TrialMatch(
                nct_id=nct_id,
                title=info.title if info else "",
                phase=info.phase if info else None,
                overall_status=info.overall_status if info else "UNKNOWN",
                locations=info.locations if info else [],
                matched_on=MatchedOn(ordinal=ordinal, text=matched_text, score=score),
                counts=VerdictCounts(
                    met=sum(v.verdict == "met" for v in verdicts),
                    not_met=sum(v.verdict == "not_met" for v in verdicts),
                    unclear=sum(v.verdict == "unclear" for v in verdicts),
                ),
                criteria=verdicts,
            )
        )

    response = MatchResponse(
        query=text,
        generated_at=datetime.now(UTC),
        model=judge.model_name,
        trials=trials,
    )
    timing = MatchTiming(
        embed_seconds=t1 - t0,
        retrieve_seconds=t2 - t1,
        judge_seconds=t3 - t2,
        cache_hits=len(nct_ids) - len(to_judge),
        judged=len(to_judge),
    )
    return response, timing
