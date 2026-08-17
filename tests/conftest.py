"""Shared pytest configuration.

Integration tests are opt-in. They hit things the rest of the suite deliberately
fakes: OpenRouter (which bills per call), ClinicalTrials.gov, and a 1.3 GB
embedding model on disk. Skipped unless ``--integration`` is passed, so the
default run stays offline, free and fast — which is also what CI needs, since
every job there is offline by design.

    uv run pytest                                     # 248 offline tests
    uv run pytest --integration                       # everything, live calls too
    uv run pytest --integration -m integration --no-cov

The last form runs only the live tests. ``--no-cov`` is needed there because the
80% gate is sized for the full suite; a dozen integration tests exercise a slice
of the codebase and would fail it for the wrong reason.

Why these exist at all: the two most expensive bugs in this project were both
OpenRouter response-shape surprises — ``content`` arriving as a list of typed
parts, and SSE keep-alive padding whose ``JSONDecodeError`` killed a run three
batches deep. Both are covered now, but by mocks written *after* observing them.
A mock can only reproduce a change someone already saw. These tests are the only
thing that can notice the third one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable

_FLAG = "--integration"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        _FLAG,
        action="store_true",
        default=False,
        help="run tests that hit live services (OpenRouter bills per call)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: Iterable[pytest.Item]) -> None:
    if config.getoption(_FLAG):
        return
    skip = pytest.mark.skip(reason=f"live-service test; pass {_FLAG} to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
