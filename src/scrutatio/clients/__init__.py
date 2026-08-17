"""Clients for the external HTTP services this project depends on.

Two of them, and they had drifted apart: ``ctgov`` was a package while the
OpenRouter transport sat loose at the package root, even though both do the same
job — hold a base URL and a credential, retry what is worth retrying, and absorb
one vendor's response quirks so the rest of the codebase does not have to.

They are grouped here rather than in ``utils`` because they are not helpers.
``utils`` is for code with no knowledge of anything outside the process;
everything in here knows about a specific remote service, its failure modes and
its bill. Keeping the two apart is also what stops ``utils`` from collecting
mismatched dependency profiles — ``utils.hashing`` is imported by the pipeline,
where neither ``fastapi`` nor an API key exists.
"""

from scrutatio.clients.openrouter import (
    OpenRouterClient,
    OpenRouterError,
    message_text,
    parse_body,
)

__all__ = [
    "OpenRouterClient",
    "OpenRouterError",
    "message_text",
    "parse_body",
]
