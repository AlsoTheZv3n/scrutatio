"""HTTP surface for the frontend.

Two constraints shape this more than anything else, and both are honest limits
rather than design choices:

**DuckDB allows one process.** While a backfill, extraction or embedding run
holds the file, nothing else can open it — not even read-only, which was
measured rather than assumed. Connections are therefore opened per request and
released immediately, so the API never blocks a pipeline run permanently, and a
request that arrives mid-run gets a 503 saying exactly that instead of hanging.

**Endpoints that have no data say so.** ``/match`` needs the Gold layer, which
needs the extraction pass to finish. Until then it returns 503 naming what is
missing and how far along it is, rather than a stub that looks like a result. The
frontend develops against the fixture in
``frontend/src/services/API/fixtures/`` and the endpoint starts working on its
own once the vectors exist.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scrutatio.config import get_settings
from scrutatio.storage import (
    bronze_count,
    bronze_max_update,
    database,
    ensure_gold,
    ensure_silver,
    ensure_storage,
    extraction_signature,
    gold_stats,
    silver_stats,
)
from scrutatio.storage.db import DatabaseBusyError
from scrutatio.storage.silver import SILVER_TABLE

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)


@contextmanager
def _open_db() -> Iterator[duckdb.DuckDBPyConnection]:
    """One connection per request; a pipeline run wins the file."""
    try:
        with database() as conn:
            yield conn
    except DatabaseBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{exc} Retry once the run finishes.",
        ) from exc


def get_db() -> Iterator[duckdb.DuckDBPyConnection]:
    with _open_db() as conn:
        yield conn


Db = Annotated["duckdb.DuckDBPyConnection", Depends(get_db)]


class Health(BaseModel):
    status: str = "ok"
    signature: str


class LayerCounts(BaseModel):
    bronze_trials: int
    newest_update: str | None
    silver_trials: int
    silver_criteria: int
    silver_failed: int
    gold_vectors: int
    embedding_model: str
    signature: str


class CriterionRow(BaseModel):
    nct_id: str
    ordinal: int
    text: str
    kind: str
    is_exclusion: bool


class CriterionPage(BaseModel):
    total: int
    offset: int
    limit: int
    rows: list[CriterionRow] = Field(default_factory=list)


class MatchRequest(BaseModel):
    text: str = Field(
        min_length=10,
        max_length=4000,
        description="Free-text patient description. Synthetic vignettes only.",
    )
    k: int = Field(default=10, ge=1, le=50)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Scrutatio",
        description=(
            "Decision support, not medical advice. Synthetic patient vignettes only — "
            "do not send real patient data."
        ),
        version="0.1.0",
    )

    # The frontend is a Vite dev server on another port, so every request is
    # cross-origin. No credentials: there are no cookies and no sessions, so
    # allowing them would widen the surface for nothing.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health", response_model=Health)
    def health() -> Health:
        """Liveness. Deliberately does not touch the database, so it still
        answers while a pipeline run holds the file."""
        return Health(signature=extraction_signature())

    @app.get("/corpus/stats", response_model=LayerCounts)
    def corpus_stats(db: Db) -> LayerCounts:
        ensure_storage(db)
        ensure_silver(db)
        ensure_gold(db)
        signature = extraction_signature()
        silver = silver_stats(db, signature)
        gold = gold_stats(db, signature, settings.embedding_model)
        newest = bronze_max_update(db)
        return LayerCounts(
            bronze_trials=bronze_count(db),
            newest_update=newest.isoformat() if newest else None,
            silver_trials=silver["trials"],
            silver_criteria=silver["criteria"],
            silver_failed=silver["failed"],
            gold_vectors=gold["vectors"],
            embedding_model=settings.embedding_model,
            signature=signature,
        )

    @app.get("/corpus/criteria", response_model=CriterionPage)
    def corpus_criteria(
        db: Db,
        kind: Annotated[str | None, Query(max_length=32)] = None,
        is_exclusion: bool | None = None,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
    ) -> CriterionPage:
        """The corpus explorer. Needs no model at request time, so it costs
        nothing per visitor and works as soon as extraction has run."""
        ensure_silver(db)
        signature = extraction_signature()

        where = ["signature = ?"]
        params: list[object] = [signature]
        if kind is not None:
            where.append("kind = ?")
            params.append(kind)
        if is_exclusion is not None:
            where.append("is_exclusion = ?")
            params.append(is_exclusion)
        clause = " AND ".join(where)

        total_row = db.execute(
            f"SELECT count(*) FROM {SILVER_TABLE} WHERE {clause}",  # noqa: S608
            params,
        ).fetchone()
        rows = db.execute(
            f"""
            SELECT nct_id, ordinal, text, kind, is_exclusion
            FROM {SILVER_TABLE} WHERE {clause}
            ORDER BY nct_id, ordinal
            LIMIT ? OFFSET ?
            """,  # noqa: S608
            [*params, limit, offset],
        ).fetchall()

        return CriterionPage(
            total=int(total_row[0]) if total_row else 0,
            offset=offset,
            limit=limit,
            rows=[
                CriterionRow(
                    nct_id=r[0], ordinal=r[1], text=r[2], kind=r[3], is_exclusion=bool(r[4])
                )
                for r in rows
            ],
        )

    @app.post("/match")
    def match(request: MatchRequest, db: Db) -> dict[str, object]:
        """Not yet available, and says why rather than returning a stub.

        Becomes real the moment the Gold layer exists — the readiness check reads
        the same counters the pipeline writes, so nothing here needs changing.
        """
        signature = extraction_signature()
        ensure_silver(db)
        ensure_gold(db)
        gold = gold_stats(db, signature, settings.embedding_model)
        if gold["vectors"] == 0:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Matching is not available yet: the Gold layer holds no vectors. "
                    f"Silver has {gold['criteria']} criteria under signature {signature}; "
                    f"run `scrutatio embed` to build the index. "
                    "Develop against frontend/src/services/API/fixtures/match-example.json."
                ),
            )
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                f"The index is ready ({gold['vectors']} vectors) but the retrieve-then-rerank "
                "pipeline is not implemented yet."
            ),
        )

    return app
