"""Stable short hashes.

Three places computed a truncated SHA-256 independently — the extraction
signature in ``storage/silver.py``, the judge prompt version in
``matching/judge.py`` and the query key in ``storage/cache.py``. Two of them were
the same operation written twice, and one of those did its ``import hashlib``
inside the function body.

That matters more than tidiness. All three feed cache keys, so a drift between
two copies does not raise — it silently splits one cache into two, or worse,
makes stale rows look current. Keeping the computation in one place is what makes
"same inputs, same key" checkable.

**These digests are load-bearing and must not change casually.** The extraction
signature keys 351,745 Silver rows: alter what goes into it, or how, and every
row reads as belonging to a different extraction and re-queues. ``ensure_ascii``
is therefore pinned here rather than left to the caller — the value below is the
one that reproduces signature ``7991e6bcb6ba4331`` on the current corpus, which
was verified before and after this helper was introduced.

These are content keys, not security primitives. Truncation is deliberate: 16 hex
characters is 64 bits, which is ample for telling a handful of prompt revisions
apart and short enough to read in a log line.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

# 64 bits for signatures, 128 for free-text queries. The wider default for text
# exists because query keys come from user input rather than from a small set of
# our own revisions, so the space of distinct values is far larger.
SIGNATURE_LENGTH: Final = 16
QUERY_LENGTH: Final = 32


def stable_digest(payload: object, *, length: int = SIGNATURE_LENGTH) -> str:
    """Deterministic short hash of a JSON-serialisable value.

    Canonicalises through ``json.dumps`` with sorted keys, so a dict written in a
    different order yields the same digest — otherwise reordering a literal in
    the source would invalidate every cached row that depended on it.

    ``ensure_ascii=False`` is not a preference: it reproduces the digest the
    corpus was built with. Flipping it changes the bytes hashed for any payload
    containing non-ASCII text.
    """
    material = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:length]


def text_digest(text: str, *, length: int = QUERY_LENGTH) -> str:
    """Stable key for a free-text input.

    Whitespace-normalised and lowercased: the same vignette pasted with a
    different line wrap is the same question, and paying a model call twice for
    it would be a surprise rather than a feature.
    """
    normalised = " ".join(text.split()).lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:length]
