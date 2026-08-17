"""Turning failures into responses the frontend can act on.

Before this module the HTTP surface handled exactly three cases inline — the
database being locked, the Gold layer being empty, and a missing API key — each
as an ad-hoc ``HTTPException``. Everything else reached the caller as FastAPI's
bare ``500 Internal Server Error``: no code to branch on, no identifier tying the
client's report to a line in the server log, and a traceback that the default
handler is happy to surface. A clinician-facing tool cannot answer "why did that
fail" with a shrug.

Three decisions shape what follows.

**Errors carry a code, not just prose.** ``detail`` is for a human; ``code`` is
what the frontend switches on. A UI that has to pattern-match English sentences
breaks the moment someone improves the wording.

**Every response carries a request id.** Generated per request, echoed in the
``X-Request-ID`` header, repeated in the error body, and logged with the
traceback. That is the difference between "a user says it broke" and "here is the
failure".

**Domain exceptions are translated explicitly, not by inheritance.** The obvious
alternative is a shared base class on every error type in the codebase. That is
the right long-term shape, and it is deliberately not done here: ``OpenRouterError``
and the extraction path's ``ExtractionError`` currently share no base beyond
``RuntimeError``, so retrofitting one touches the extraction transport and its 39
tests. An explicit table is honest about the hierarchy as it actually is, and it
fails loudly — an unmapped type lands in the catch-all and is logged — rather
than silently inheriting the wrong status.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Final

import duckdb
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from scrutatio.clients.openrouter import OpenRouterError
from scrutatio.embeddings import EncoderUnavailableError
from scrutatio.storage.db import DatabaseBusyError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER: Final = "X-Request-ID"

# Short enough to read back over the phone, wide enough not to collide within a
# log retention window.
_REQUEST_ID_LENGTH: Final = 12


class ErrorBody(BaseModel):
    """The shape of every non-2xx response from this API.

    Mirrored in ``frontend/src/services/API/types.ts`` — a second definition that
    has to be kept in step by hand, which is why the field set is small.
    """

    code: str = Field(description="Stable machine-readable identifier. Switch on this.")
    detail: str = Field(description="Human-readable explanation. Do not parse.")
    request_id: str = Field(description="Correlates this response with the server log.")


class ApiError(Exception):
    """Raised by endpoints that know both the status and the reason.

    Preferred over ``HTTPException`` because it forces a ``code``. An endpoint
    that cannot name the failure has not finished thinking about it.
    """

    def __init__(self, *, status_code: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail


# Ordered: the first matching type wins, so any subclass must precede its base.
# Nothing here is a subclass of anything else at present — DatabaseBusyError is a
# RuntimeError, not a duckdb.Error — but the order is load-bearing if that changes.
_TRANSLATIONS: Final[tuple[tuple[type[Exception], int, str], ...]] = (
    (
        DatabaseBusyError,
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "database_busy",
    ),
    (
        EncoderUnavailableError,
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "encoder_unavailable",
    ),
    # 502 rather than 500: the fault is upstream at OpenRouter, and the
    # distinction tells the caller whether retrying is sensible.
    (
        OpenRouterError,
        status.HTTP_502_BAD_GATEWAY,
        "upstream_failed",
    ),
    (
        duckdb.Error,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "storage_error",
    ),
)


def request_id(request: Request) -> str:
    """The current request's id, or a fresh one if the middleware never ran.

    The fallback exists because exception handlers must never be the thing that
    raises. A handler failing on a missing attribute would replace a diagnosable
    502 with an undiagnosable 500.
    """
    return getattr(request.state, "request_id", None) or uuid.uuid4().hex[:_REQUEST_ID_LENGTH]


def _body(
    request: Request,
    *,
    code: str,
    detail: str,
    status_code: int,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """One error shape, plus whatever headers the status itself requires.

    ``headers`` exists because reshaping a framework response silently discarded
    the ones it had set. For 405 that is not cosmetic: RFC 9110 §15.5.6 says the
    server MUST send ``Allow``, and without it the client is told "no" with no way
    to learn what would have worked.
    """
    rid = request_id(request)
    return JSONResponse(
        status_code=status_code,
        content=ErrorBody(code=code, detail=detail, request_id=rid).model_dump(),
        headers={**(headers or {}), REQUEST_ID_HEADER: rid},
    )


def install_error_handling(app: FastAPI) -> None:
    """Attach request ids and the exception handlers to an app.

    Call this *before* adding CORS. Starlette runs the last-added middleware
    outermost, so CORS added afterwards wraps this one — which is what puts CORS
    headers on error responses too. A browser cannot read a 502 it is not allowed
    to see, and the frontend would report a network failure instead of the
    upstream failure that actually happened.
    """

    @app.middleware("http")
    async def _tag_request(
        request: Request, call_next: Callable[[Request], Awaitable[JSONResponse]]
    ) -> JSONResponse:
        rid = uuid.uuid4().hex[:_REQUEST_ID_LENGTH]
        request.state.request_id = rid
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = rid
        return response

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        # Expected and named, so info rather than exception: no traceback wanted.
        logger.info("%s -> %d %s: %s", request.url.path, exc.status_code, exc.code, exc.detail)
        return _body(request, code=exc.code, detail=exc.detail, status_code=exc.status_code)

    def _register(exc_type: type[Exception], status_code: int, code: str) -> None:
        async def handler(request: Request, exc: Exception) -> JSONResponse:
            logger.warning("%s -> %d %s: %s", request.url.path, status_code, code, str(exc)[:300])
            return _body(request, code=code, detail=str(exc), status_code=status_code)

        app.add_exception_handler(exc_type, handler)  # type: ignore[arg-type]

    for exc_type, status_code, code in _TRANSLATIONS:
        _register(exc_type, status_code, code)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Covers what the framework raises on its own — 404 for an unknown route,
        # 405 for the wrong verb. Reshaped so there is exactly one error format,
        # but the framework's headers are carried through rather than dropped.
        headers = exc.headers or {}
        detail = str(exc.detail)

        allow = next((v for k, v in headers.items() if k.lower() == "allow"), None)
        if allow and exc.status_code == status.HTTP_405_METHOD_NOT_ALLOWED:
            # "Method Not Allowed" alone answers nothing, and the usual way to meet
            # this is a browser address bar on a POST-only route. Say what works.
            detail = f"{detail}. {request.url.path} accepts: {allow}."

        return _body(
            request,
            code=f"http_{exc.status_code}",
            detail=detail,
            status_code=exc.status_code,
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Field paths flattened into one line. The structured form is richer, but
        # it would mean a second response shape for the frontend to handle.
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg', '')}".strip(": ")
            for e in exc.errors()
        )
        return _body(
            request,
            code="validation_error",
            detail=problems or "The request body did not match the expected shape.",
            # Not _ENTITY: Starlette deprecated that spelling in favour of
            # _CONTENT, which is the name RFC 9110 settled on. Caught by the
            # suite's strict warning filter rather than by review.
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # The only handler that logs a traceback, and the only one whose detail is
        # deliberately uninformative: an unmapped failure may carry a connection
        # string or a prompt fragment in its message, and that is not the
        # caller's to see. The request id is how the two halves are joined.
        rid = request_id(request)
        logger.exception(
            "Unhandled %s on %s [request_id=%s]",
            type(exc).__name__,
            request.url.path,
            rid,
        )
        return _body(
            request,
            code="internal_error",
            detail=f"An unexpected error occurred. Quote request id {rid} when reporting it.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
