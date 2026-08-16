"""Encoder tests — the two failure modes that are worth catching early.

The encoder is a thin wrapper over sentence-transformers, which is an optional
dependency because it pulls torch. These tests therefore mock the library rather
than install it: what is being checked is our handling, not theirs.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from scrutatio.config import EMBEDDING_DIMENSIONS
from scrutatio.embeddings import CriterionEncoder, EncoderUnavailableError


class TestAvailability:
    def test_the_package_imports_without_the_embedding_stack(self) -> None:
        # torch is a large optional dependency; the rest of the pipeline must not
        # require it to be installed.
        assert CriterionEncoder().model_name.startswith("BAAI/bge-")

    def test_a_missing_library_says_how_to_install_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        with pytest.raises(EncoderUnavailableError, match="--extra embeddings"):
            _ = CriterionEncoder().model


class TestDimensionGuard:
    def _fake_library(self, monkeypatch: pytest.MonkeyPatch, dims: int) -> None:
        class FakeModel:
            def __init__(self, *_: Any, **__: Any) -> None: ...

            def get_sentence_embedding_dimension(self) -> int:
                return dims

        monkeypatch.setitem(
            sys.modules,
            "sentence_transformers",
            SimpleNamespace(SentenceTransformer=FakeModel),
        )

    def test_a_mismatched_model_is_rejected_at_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The Gold column is FLOAT[1024]. Without this guard a wrong model only
        # fails on the first insert — thousands of encodes into a run.
        self._fake_library(monkeypatch, dims=768)
        with pytest.raises(ValueError, match=f"FLOAT\\[{EMBEDDING_DIMENSIONS}\\]"):
            _ = CriterionEncoder(device="cpu").model

    def test_a_matching_model_loads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fake_library(monkeypatch, dims=EMBEDDING_DIMENSIONS)
        assert CriterionEncoder(device="cpu").model is not None

    def test_the_model_is_loaded_once_and_reused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._fake_library(monkeypatch, dims=EMBEDDING_DIMENSIONS)
        encoder = CriterionEncoder(device="cpu")
        assert encoder.model is encoder.model


class TestEmptyInput:
    def test_encoding_nothing_does_not_load_the_model(self) -> None:
        # Loading BGE is 1.3 GB and several seconds; an empty batch must not pay
        # for it. This also keeps the run loop's final empty iteration cheap.
        assert CriterionEncoder().encode_criteria([]) == []
