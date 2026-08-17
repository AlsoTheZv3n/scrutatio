"""The local halves that the offline suite fakes: a real file lock, a real model.

No network and no credentials, but these are integration tests all the same —
they exercise a real OS file lock and a real 1.3 GB embedding model instead of a
monkeypatched stand-in.

Both cover claims the architecture rests on and nothing verifies:

* The API returns 503 during a pipeline run because DuckDB permits one process.
  The offline test proves the *handler* works by making ``database`` raise. It has
  never proved the lock exists.
* Embeddings are 1024-dimensional and BGE is asymmetric. The offline tests use
  constructed vectors of the right shape, which agree with the code by
  construction.

Never against ``data/scrutatio.duckdb``: it is 3.2 GB of real corpus, and a test
that locks it would collide with any running pipeline. Temp files only.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from scrutatio.config import EMBEDDING_DIMENSIONS
from scrutatio.storage.bronze import ensure_storage
from scrutatio.storage.db import IN_MEMORY, connect
from scrutatio.storage.gold import ensure_gold, search_criteria, write_embeddings
from scrutatio.storage.silver import SILVER_TABLE, ensure_silver

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration

_SIG = "integration-sig"

# Table name is a module constant, so S608 is a false positive here — same as at
# every other call site in the codebase.
_INSERT = f"INSERT INTO {SILVER_TABLE} VALUES (?,?,0,NULL,?,'condition',false,now())"  # noqa: S608


def _probe(target: Path, *, read_only: bool = False) -> str:
    """Try to open ``target`` from a *separate process* and report what happened.

    A subprocess is not ceremony. The first version of this test used a second
    connection in the same process and did not raise at all: DuckDB's exclusion is
    between processes, not between connections. Getting that wrong produced a test
    that asserted nothing about the behaviour the API depends on.
    """
    if read_only:
        source = (
            "import duckdb, sys\n"
            "try:\n"
            "    duckdb.connect(sys.argv[1], read_only=True)\n"
            "except Exception as exc:\n"
            "    print(type(exc).__name__)\n"
            "else:\n"
            "    print('OPENED')\n"
        )
    else:
        source = (
            "import sys\n"
            "from scrutatio.storage.db import DatabaseBusyError, connect\n"
            "try:\n"
            "    connect(sys.argv[1])\n"
            "except DatabaseBusyError as exc:\n"
            "    print('DatabaseBusyError'); print(exc)\n"
            "except Exception as exc:\n"
            "    print(type(exc).__name__); print(exc)\n"
            "else:\n"
            "    print('OPENED')\n"
        )
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-c", source, str(target)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return done.stdout.strip() or f"<no output; stderr: {done.stderr.strip()[:200]}>"


class TestTheSingleWriterLockIsReal:
    """DuckDB permits one read-write *process* and refuses every other one.

    The whole API design follows from this — per-request connections, a 503 while a
    run holds the file, and it is the strongest argument for moving to PostgreSQL.
    It had been measured once by hand; this makes it a standing check.

    Measured here, on a German-locale Windows machine, because it matters for how
    ``db.py`` recognises the condition. The message is two-part:

        IO Error: Cannot open file "...": Der Prozess kann nicht auf die Datei
        zugreifen, da sie von einem anderen Prozess verwendet wird.
        File is already open in D:\\dev\\python.exe (PID 49612)

    The first half is the localised OS error; only the second is DuckDB's own
    wording. That is exactly why ``_LOCK_MARKER`` matches on "File is already
    open" and not on anything the OS produced.
    """

    def test_a_second_process_is_refused_as_a_busy_database(self, tmp_path: Path) -> None:
        target = tmp_path / "locked.duckdb"
        held = connect(target)
        try:
            result = _probe(target)
        finally:
            held.close()

        assert result.startswith("DatabaseBusyError"), result
        # The message is what the API hands the caller, so it has to name the cause
        # rather than leak DuckDB's IOException.
        assert "another process" in result

    def test_the_marker_survives_a_localised_os_message(self, tmp_path: Path) -> None:
        # The half of the message that db.py must NOT depend on is localised; the
        # half it does depend on is not. If DuckDB ever drops its own note, the
        # wrapper silently stops converting and the API answers 500 instead of 503.
        target = tmp_path / "locked.duckdb"
        held = connect(target)
        try:
            import duckdb

            probe = (
                "import duckdb, sys\n"
                "try:\n"
                "    duckdb.connect(sys.argv[1])\n"
                "except duckdb.IOException as exc:\n"
                "    print(str(exc))\n"
            )
            done = subprocess.run(  # noqa: S603
                [sys.executable, "-c", probe, str(target)],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        finally:
            held.close()

        assert "File is already open" in done.stdout, done.stdout
        assert duckdb.__version__  # the version this was measured against

    def test_read_only_is_refused_too(self, tmp_path: Path) -> None:
        # The surprising half, and the reason a UI cannot simply read the live file
        # while a run writes it. Asserted because it is counter-intuitive enough
        # that someone will eventually assume the opposite and design around it.
        target = tmp_path / "locked.duckdb"
        held = connect(target)
        try:
            result = _probe(target, read_only=True)
        finally:
            held.close()

        assert result == "IOException", result

    def test_two_connections_in_a_single_process_coexist(self, tmp_path: Path) -> None:
        # The other side of the same coin, and load-bearing for the API: it opens a
        # connection per request, and those must not fight each other. Learned by
        # getting the test above wrong first.
        target = tmp_path / "shared.duckdb"
        first = connect(target)
        second = connect(target)
        try:
            first.execute("CREATE TABLE t (x INTEGER)")
            second.execute("INSERT INTO t VALUES (1)")
            assert second.execute("SELECT count(*) FROM t").fetchone() == (1,)
        finally:
            first.close()
            second.close()

    def test_the_lock_is_released_on_close(self, tmp_path: Path) -> None:
        target = tmp_path / "released.duckdb"
        connect(target).close()
        assert _probe(target) == "OPENED"


@pytest.fixture(scope="module")
def encoder() -> object:
    """The real model, loaded once for the module.

    Module-scoped and defined at module level rather than as a class-scoped
    instance method — pytest deprecated the latter, and the suite's strict warning
    filter turns that into an error.
    """
    pytest.importorskip("sentence_transformers", reason="the `embeddings` extra is not installed")
    from scrutatio.embeddings import CriterionEncoder

    return CriterionEncoder()


class TestTheRealEncoder:
    """Loads the actual model. Slow and worth it once per integration run."""

    def test_the_model_produces_the_dimension_gold_expects(self, encoder: object) -> None:
        # The guard in `_load` raises on a mismatch, so reaching this line already
        # proves it. Asserted anyway: the Gold column is FLOAT[1024], and a
        # mismatch would otherwise surface thousands of vectors into a run.
        vectors = encoder.encode_criteria(["ECOG performance status 0 or 1"])
        assert vectors.shape == (1, EMBEDDING_DIMENSIONS)
        assert vectors.dtype.name == "float32"

    def test_the_query_prefix_is_actually_applied(self, encoder: object) -> None:
        # BGE is trained with an instruction prefix on the query side only.
        # Embedding both sides the same way is a silent retrieval-quality loss, and
        # until now that asymmetry was only a comment.
        import numpy as np

        text = "patient with metastatic lung cancer"
        as_query = encoder.encode_query(text)
        as_document = encoder.encode_criteria([text])[0]
        assert not np.allclose(as_query, as_document), "query and document sides are identical"

    def test_device_selection_reports_something_usable(self, encoder: object) -> None:
        # `_pick_device` silently choosing CPU cost ~22 hours of embedding time.
        # This does not demand CUDA — CI has none — only that the choice is a real
        # device and visible in the record.
        assert encoder._pick_device() in {"cuda", "cpu"}


class TestRealVectorsRoundTrip:
    """Encode, bulk-load through Arrow, search — with real vectors.

    The offline Gold tests construct vectors themselves, so they agree with the
    schema by construction. This is the only path that would catch a dtype or
    dimension mismatch between what the encoder emits and what Arrow writes.
    """

    def test_a_semantic_query_ranks_the_right_criterion_first(self) -> None:
        pytest.importorskip(
            "sentence_transformers", reason="the `embeddings` extra is not installed"
        )
        from scrutatio.embeddings import CriterionEncoder

        criteria = [
            ("NCT00000001", "Histologically confirmed stage IV non-small cell lung cancer"),
            ("NCT00000002", "New York Heart Association class III or IV heart failure"),
            ("NCT00000003", "Written informed consent obtained before any study procedure"),
        ]

        encoder = CriterionEncoder()
        db = connect(IN_MEMORY)
        try:
            ensure_storage(db)
            ensure_silver(db)
            ensure_gold(db)
            for identifier, text in criteria:
                db.execute(_INSERT, [identifier, _SIG, text])

            vectors = encoder.encode_criteria([t for _, t in criteria])
            written = write_embeddings(
                db,
                [(identifier, 0, vectors[i]) for i, (identifier, _) in enumerate(criteria)],
                signature=_SIG,
                model=encoder.model_name,
            )
            assert written == 3

            hits = search_criteria(
                db,
                encoder.encode_query("metastatic non-small cell lung carcinoma"),
                signature=_SIG,
                model=encoder.model_name,
                k=3,
            )
        finally:
            db.close()

        assert len(hits) == 3
        # Real embeddings, real cosine. If the lung criterion is not first, either
        # the vectors are not what we think or the similarity query is wrong.
        assert hits[0][0] == "NCT00000001", [(h[0], round(h[1], 3)) for h in hits]
        # And scores must be ordered, since the API trusts that ordering.
        assert hits[0][1] >= hits[1][1] >= hits[2][1]
