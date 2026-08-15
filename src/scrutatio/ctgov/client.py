"""ClinicalTrials.gov API v2 client.

The API is public: no key, no registration, no required User-Agent, and no
published rate limit. That last point cuts both ways — there is nothing to
design against, so this client stays deliberately polite (bounded concurrency
upstream, exponential backoff on 429/5xx).

Three behaviours of the API are easy to get wrong and are handled here:

1. **Paging must repeat the query.** Every parameter except ``countTotal``,
   ``pageSize`` and ``pageToken`` has to be resent on each page. Dropping them
   silently re-queries the whole registry.
2. **Unknown parameters are a hard 400**, not an ignored extra.
3. **The last-update date is nested.** It lives at
   ``protocolSection.statusModule.lastUpdatePostDateStruct.date``. Reading a
   flat ``lastUpdatePostDate`` yields ``None`` for every study, which freezes an
   incremental watermark without raising anything.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from datetime import date
from typing import Any, Final

import httpx
from pydantic import ValidationError

from scrutatio.config import Settings, get_settings
from scrutatio.ctgov.models import PagedStudies, Study

logger = logging.getLogger(__name__)

# Coerced down to this by the API anyway; enforced locally so the value we send
# is the value we get.
MAX_PAGE_SIZE: Final = 1000

# Parameters that must NOT be repeated on follow-up pages.
_PAGING_ONLY: Final = frozenset({"countTotal", "pageSize", "pageToken"})

_RETRY_STATUS: Final = frozenset({429, 500, 502, 503, 504})


class CtGovError(RuntimeError):
    """A request to ClinicalTrials.gov failed after exhausting retries."""


def nct_id(study: Study) -> str | None:
    """Extract the NCT identifier as a plain string.

    The generated model wraps it in a ``RootModel``, so the raw attribute is an
    ``Nct`` object rather than a ``str`` — comparing it to a string silently
    fails. Every consumer wants the string.
    """
    protocol = study.protocol_section
    if protocol is None or protocol.identification_module is None:
        return None
    wrapped = protocol.identification_module.nct_id
    return None if wrapped is None else str(wrapped.root)


def last_update_posted(study: Study) -> date | None:
    """Extract the incremental watermark from a study.

    Returns ``None`` when the study carries no last-update date, which is rare
    but permitted by the schema.
    """
    protocol = study.protocol_section
    if protocol is None or protocol.status_module is None:
        return None
    struct = protocol.status_module.last_update_post_date_struct
    if struct is None:
        return None
    return struct.date


class CtGovClient:
    """Synchronous client for the studies endpoint.

    Usable as a context manager; owns its ``httpx.Client`` unless one is
    injected (which the tests do).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._base_url = str(self._settings.ctgov_base_url).rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=self._settings.request_timeout_seconds,
            headers={"Accept": "application/json"},
            follow_redirects=True,
        )

    def __enter__(self) -> CtGovClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # -- low level ---------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET with exponential backoff on transient failures."""
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None

        for attempt in range(self._settings.max_retries + 1):
            try:
                response = self._client.get(url, params=params)
            except httpx.HTTPError as exc:  # network-level
                last_exc = exc
            else:
                if response.status_code == httpx.codes.OK:
                    return response.json()
                if response.status_code not in _RETRY_STATUS:
                    # 400 means a malformed query — retrying cannot help.
                    msg = (
                        f"ClinicalTrials.gov returned {response.status_code} for {path}: "
                        f"{response.text[:300]}"
                    )
                    raise CtGovError(msg)
                last_exc = CtGovError(f"HTTP {response.status_code}")

            if attempt < self._settings.max_retries:
                delay = 2.0**attempt
                logger.warning(
                    "ClinicalTrials.gov request failed (attempt %d/%d), retrying in %.0fs: %s",
                    attempt + 1,
                    self._settings.max_retries + 1,
                    delay,
                    last_exc,
                )
                time.sleep(delay)

        attempts = self._settings.max_retries + 1
        msg = f"ClinicalTrials.gov request to {path} failed after {attempts} attempts"
        raise CtGovError(msg) from last_exc

    # -- public ------------------------------------------------------------

    def count(self, *, query: dict[str, Any] | None = None) -> int:
        """Total number of studies matching the scope (or an explicit query)."""
        params = {**(query or self._scope_query()), "countTotal": "true", "pageSize": 1}
        payload = self._get("/studies", params)
        return int(payload.get("totalCount", 0))

    def iter_studies(
        self,
        *,
        query: dict[str, Any] | None = None,
        updated_since: date | None = None,
        limit: int | None = None,
    ) -> Iterator[Study]:
        """Yield studies page by page, following ``nextPageToken``.

        ``updated_since`` narrows to studies whose last-update date is on or
        after the given day — the incremental path.
        """
        base = dict(query or self._scope_query())
        if updated_since is not None:
            base["filter.advanced"] = self._with_update_range(
                base.get("filter.advanced"), updated_since
            )

        # Guard against a caller smuggling paging keys into the query.
        base = {k: v for k, v in base.items() if k not in _PAGING_ONLY}

        page_size = min(self._settings.page_size, MAX_PAGE_SIZE)
        token: str | None = None
        yielded = 0

        while True:
            params = {**base, "pageSize": page_size}
            if token is not None:
                params["pageToken"] = token

            payload = self._get("/studies", params)

            try:
                page = PagedStudies.model_validate(payload)
            except ValidationError as exc:
                msg = f"Response did not match the generated schema: {exc}"
                raise CtGovError(msg) from exc

            # `studies` is a RootModel wrapper around list[Study], not a plain list.
            for study in page.studies.root:
                yield study
                yielded += 1
                if limit is not None and yielded >= limit:
                    return

            token = page.next_page_token
            if not token:
                return

    def fetch_study(self, nct_id: str) -> Study:
        """Fetch a single study by NCT ID."""
        payload = self._get(f"/studies/{nct_id}", {})
        try:
            return Study.model_validate(payload)
        except ValidationError as exc:
            msg = f"Study {nct_id} did not match the generated schema: {exc}"
            raise CtGovError(msg) from exc

    # -- query construction ------------------------------------------------

    def _scope_query(self) -> dict[str, Any]:
        """The configured oncology scope as request parameters."""
        params: dict[str, Any] = {"filter.advanced": self._settings.scope_filter}
        if self._settings.recruiting_only:
            params["filter.overallStatus"] = "RECRUITING"
        return params

    @staticmethod
    def _with_update_range(existing: str | None, since: date) -> str:
        """Append a last-update-date range to an Essie expression."""
        clause = f"AREA[LastUpdatePostDate]RANGE[{since.isoformat()},MAX]"
        return f"{existing} AND {clause}" if existing else clause
