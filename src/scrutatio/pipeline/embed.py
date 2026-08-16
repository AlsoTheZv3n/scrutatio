"""Gold run: criteria in, vectors out.

Resumable on the same principle as the other two layers — each batch is
committed before the next is fetched, and pending work is re-derived from the
tables rather than tracked in memory. Unlike extraction this is local and cheap,
so the batches are large and there is no failure table: an embedding either
computes or the process is broken.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scrutatio.embeddings.encoder import CriterionEncoder
from scrutatio.storage.gold import ensure_gold, gold_stats, pending_criteria, write_embeddings
from scrutatio.storage.silver import extraction_signature

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

# Large: the work is local and the round trip is a GPU call, not a network hop.
DEFAULT_BATCH_SIZE = 2048


@dataclass(frozen=True)
class EmbedRunResult:
    """What one embedding pass accomplished."""

    signature: str
    model: str
    written: int
    remaining: int


def run_embedding(
    *,
    db: duckdb.DuckDBPyConnection,
    encoder: CriterionEncoder,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EmbedRunResult:
    """Embed pending criteria into Gold, committing batch by batch."""
    ensure_gold(db)
    signature = extraction_signature()
    model = encoder.model_name
    logger.info("Embedding signature %s with %s", signature, model)

    written = 0
    while limit is None or written < limit:
        remaining_budget = None if limit is None else limit - written
        take = batch_size if remaining_budget is None else min(batch_size, remaining_budget)

        pending = pending_criteria(db, signature, model, limit=take)
        if not pending:
            break

        vectors = encoder.encode_criteria([text for _, _, text in pending])
        rows = [
            (nct, ordinal, vec) for (nct, ordinal, _), vec in zip(pending, vectors, strict=True)
        ]
        written += write_embeddings(db, rows, signature=signature, model=model)
        logger.info("Embedded %d criteria (%d so far)", len(rows), written)

    progress = gold_stats(db, signature, model)
    return EmbedRunResult(
        signature=signature,
        model=model,
        written=written,
        remaining=progress["criteria"] - progress["vectors"],
    )
