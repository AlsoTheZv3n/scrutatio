"""Cross-cutting helpers.

Deliberately narrow. A ``utils`` package earns its place only while everything in
it is used from more than one caller — the moment something here has a single
consumer it belongs next to that consumer instead. Two things qualify today:

* ``hashing`` — three modules computed truncated SHA-256 digests independently,
  two of them the identical operation. They all feed cache keys, where a silent
  drift between copies splits one cache into two.
* ``errors`` — the HTTP surface's failure translation.

**Only ``hashing`` is re-exported here, and that is not an oversight.**
``errors`` imports FastAPI, which lives in the optional ``serve`` extra. Naming it
in this file would execute that import for anyone touching ``scrutatio.utils`` at
all — including ``storage/silver.py``, which runs in the pipeline where FastAPI is
not installed. Importing a submodule runs the package ``__init__`` first, so the
mistake would not be contained. Reach the error layer as
``from scrutatio.utils.errors import ...`` instead.
"""

from scrutatio.utils.hashing import QUERY_LENGTH, SIGNATURE_LENGTH, stable_digest, text_digest

__all__ = [
    "QUERY_LENGTH",
    "SIGNATURE_LENGTH",
    "stable_digest",
    "text_digest",
]
