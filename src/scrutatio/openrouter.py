"""The transport shared by extraction and matching.

Both talk to the same endpoint under the same conditions, and every rule in here
was paid for once already. Duplicating them would mean the next lesson lands in
one caller and not the other.

What is encoded here, and why:

* **Keep-alive padding.** OpenRouter emits SSE-style comment lines while an
  upstream provider is slow, even on non-streaming requests. A bare
  ``response.json()`` raises on them — and that ``JSONDecodeError`` killed an
  extraction pass three batches deep, because it is not the error type the
  caller was catching.
* **Provider routing.** 18 providers serve the model in use, and default routing
  spreads requests across them. A batch finishes when its slowest concurrent
  request does, so one slow provider paces the whole wave: measured, sustained
  throughput went from ~350 to ~1,650 trials/hour by pinning to throughput.
* **Reasoning off.** For schema-constrained output, thinking tokens buy nothing
  and are billed and timed like any other output. Measured 2.5-3.8x fewer output
  tokens with the same criterion count.
* **``Retry-After`` is honoured** when the endpoint sends it, rather than backing
  off blindly against a number the server already told us.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Final

import httpx

from scrutatio.config import Settings, get_settings

logger = logging.getLogger(__name__)

_RETRY_STATUS: Final = frozenset({429, 500, 502, 503, 504})


class OpenRouterError(RuntimeError):
    """The endpoint failed, or returned something that is not usable output."""


def parse_body(response: httpx.Response) -> dict[str, Any]:
    """Parse a 200 response, tolerating keep-alive padding ahead of the JSON."""
    try:
        return response.json()  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        text = response.text
        start = text.find("{")
        if start > 0:
            try:
                parsed = json.loads(text[start:])
            except json.JSONDecodeError:
                pass
            else:
                logger.warning("Discarded %d bytes of padding before the JSON body", start)
                return parsed  # type: ignore[no-any-return]
        # The head verbatim: the next occurrence should be diagnosable without
        # reproducing it.
        msg = f"Response body was not JSON. First 200 bytes: {text[:200]!r}"
        raise OpenRouterError(msg) from None


def message_text(message: dict[str, Any]) -> str:
    """Pull the assistant text out of either response shape.

    Some models return ``content`` as a list of typed parts — a reasoning part
    followed by a text part — rather than a string. A client that assumes ``str``
    breaks on them.
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        ]
        if parts:
            return "".join(parts)
    msg = f"Could not find assistant text in message of type {type(content).__name__}"
    raise OpenRouterError(msg)


class OpenRouterClient:
    """A thin, retrying JSON client for chat completions."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        if not self._settings.openrouter_configured:
            msg = "OPENROUTER_API_KEY must be set"
            raise ValueError(msg)
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=self._settings.llm_timeout_seconds)

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @property
    def _headers(self) -> dict[str, str]:
        key = self._settings.openrouter_api_key
        assert key is not None  # guaranteed by openrouter_configured
        return {
            "Authorization": f"Bearer {key.get_secret_value()}",
            "Content-Type": "application/json",
            # Attribution; harmless and polite.
            "HTTP-Referer": "https://github.com/AlsoTheZv3n/scrutatio",
            "X-Title": "scrutatio",
        }

    def build_payload(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        reasoning: bool = False,
    ) -> dict[str, Any]:
        """Assemble a request with the defaults both callers want."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if not reasoning:
            payload["reasoning"] = {"enabled": False}
        if self._settings.extraction_provider_sort:
            payload["provider"] = {"sort": self._settings.extraction_provider_sort}
        return payload

    def post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send the request, retrying transient failures. Returns the parsed body."""
        url = self._settings.chat_completions_url
        last: str | None = None

        for attempt in range(self._settings.max_retries + 1):
            response: httpx.Response | None = None
            try:
                response = self._client.post(url, json=payload, headers=self._headers)
            except httpx.HTTPError as exc:
                last = str(exc)
            else:
                if response.status_code == httpx.codes.OK:
                    return parse_body(response)
                if response.status_code not in _RETRY_STATUS:
                    msg = f"OpenRouter returned {response.status_code}: {response.text[:300]}"
                    raise OpenRouterError(msg)
                last = f"HTTP {response.status_code}"

            if attempt < self._settings.max_retries:
                delay = self._retry_delay(response, attempt)
                logger.warning("Retry %d in %.0fs: %s", attempt + 1, delay, last)
                time.sleep(delay)

        msg = f"Request failed after {self._settings.max_retries + 1} attempts: {last}"
        raise OpenRouterError(msg)

    @staticmethod
    def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
        """Honour ``Retry-After`` when the endpoint sends it; else back off."""
        if response is not None:
            header = response.headers.get("retry-after")
            if header:
                try:
                    return float(header)
                except ValueError:
                    pass
        return 2.0**attempt
