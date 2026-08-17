"""Local embedding of criteria — no API, no per-vector cost.

``sentence-transformers`` pulls torch, which is a large dependency that CI has
no use for. It therefore lives in the optional ``embeddings`` group and is
imported on first use rather than at module import, so the rest of the package
keeps working without it.

BGE is asymmetric: the query side takes an instruction prefix, the document side
does not. Encoding both the same way is a silent quality loss, so the two paths
are separate methods rather than a flag.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from scrutatio.config import BGE_QUERY_PREFIX, EMBEDDING_DIMENSIONS, Settings, get_settings

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "sentence-transformers is not installed. It is an optional dependency because "
    "it pulls torch: `uv sync --extra embeddings`"
)


class EncoderUnavailableError(RuntimeError):
    """The embedding stack is not installed."""


class CriterionEncoder:
    """Encodes criteria and patient text into the same vector space."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.model_name = model_name or self._settings.embedding_model
        self._device = device
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        """The loaded model. Loading is deferred until something is encoded."""
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise EncoderUnavailableError(_INSTALL_HINT) from exc

        device = self._device or self._pick_device()
        logger.info("Loading %s on %s", self.model_name, device)
        model = SentenceTransformer(self.model_name, device=device)

        dims = model.get_embedding_dimension()
        if dims != EMBEDDING_DIMENSIONS:
            # The Gold column is FLOAT[1024]. A mismatch would only surface as an
            # insert failure thousands of vectors in.
            msg = (
                f"{self.model_name} produces {dims}-dimensional vectors, but the "
                f"Gold column is FLOAT[{EMBEDDING_DIMENSIONS}]"
            )
            raise ValueError(msg)
        return model

    @staticmethod
    def _pick_device() -> str:
        try:
            import torch
        except ImportError:  # pragma: no cover - depends on the environment
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def encode_criteria(self, texts: Sequence[str], *, batch_size: int = 128) -> Any:
        """Encode criterion text — the document side, no prefix.

        Returns the numpy array as produced, not a list of lists. Converting
        1,024 floats per row into Python objects only to hand them straight to a
        bulk loader was pure cost.
        """
        if len(texts) == 0:
            import numpy as np

            return np.empty((0, EMBEDDING_DIMENSIONS), dtype=np.float32)
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def encode_query(self, text: str) -> list[float]:
        """Encode a patient description — the query side, with the BGE prefix."""
        vector = self.model.encode(
            BGE_QUERY_PREFIX + text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return list(vector.tolist())
