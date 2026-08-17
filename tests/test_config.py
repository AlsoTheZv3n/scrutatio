"""Config tests, including the security property that matters most:
an API key must never surface in a repr, log line, or traceback.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scrutatio.config import (
    DEFAULT_EXTRACTION_MODEL,
    DEFAULT_SCOPE_FILTER,
    EMBEDDING_DIMENSIONS,
    Settings,
    get_settings,
    project_root,
)

DUMMY_KEY = "fake-key-for-tests-only"


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate from the developer's real .env and shell environment."""
    for var in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "EXTRACTION_MODEL",
        "MATCHING_MODEL",
        "EMBEDDING_MODEL",
        "DB_PATH",
        "SCOPE_FILTER",
        "PAGE_SIZE",
        "MAX_SPEND_USD",
        "EXTRACTION_WORKERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _settings(**kwargs: object) -> Settings:
    """Build settings without reading the on-disk .env."""
    return Settings(_env_file=None, **kwargs)  # type: ignore[arg-type]


class TestDefaults:
    def test_scope_defaults_to_interventional_oncology(self, clean_env: None) -> None:
        assert _settings().scope_filter == DEFAULT_SCOPE_FILTER
        assert "Neoplasms" in DEFAULT_SCOPE_FILTER
        assert "INTERVENTIONAL" in DEFAULT_SCOPE_FILTER

    def test_extraction_and_matching_default_to_the_same_model(self, clean_env: None) -> None:
        s = _settings()
        assert s.extraction_model == DEFAULT_EXTRACTION_MODEL
        assert s.matching_model == DEFAULT_EXTRACTION_MODEL

    def test_default_model_is_namespaced_for_openrouter(self, clean_env: None) -> None:
        # OpenRouter addresses models as "provider/model"; a bare name 404s.
        assert "/" in _settings().extraction_model

    def test_default_embedding_model_needs_no_remote_code(self, clean_env: None) -> None:
        # The long-context model (gte-large-en-v1.5) was chosen when whole
        # eligibility texts were to be embedded — a mean of ~835 tokens against
        # BGE's 512-token window. We embed *criteria*, which measure ~31 tokens,
        # so the window stopped being the deciding factor and what remained was
        # that gte requires trust_remote_code=True: building the index would
        # execute code fetched from a model repository.
        assert _settings().embedding_model.startswith("BAAI/bge-")

    def test_embedding_dimensions_match_measured_value(self) -> None:
        assert EMBEDDING_DIMENSIONS == 1024

    def test_recruiting_only_by_default(self, clean_env: None) -> None:
        assert _settings().recruiting_only is True

    def test_database_is_a_local_file(self, clean_env: None) -> None:
        assert _settings().db_path.suffix == ".duckdb"

    def test_the_database_path_is_absolute(self, clean_env: None) -> None:
        # A relative path plus connect()'s mkdir(parents=True) means running from
        # the wrong directory does not fail — it creates an empty database. That
        # happened: the API was started from src/scrutatio/, reported zero for every
        # layer, and looked exactly like a lost 3.2 GB corpus.
        assert _settings().db_path.is_absolute()

    def test_the_database_path_does_not_depend_on_the_working_directory(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from_here = _settings().db_path
        monkeypatch.chdir(tmp_path)
        assert _settings().db_path == from_here

    def test_it_is_anchored_to_the_checkout(self, clean_env: None) -> None:
        root = project_root()
        assert root is not None, "these tests run from a checkout"
        assert _settings().db_path == root / "data" / "scrutatio.duckdb"

    def test_the_env_file_is_found_from_any_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Same trap as db_path, and worse: a .env resolved against the working
        # directory means starting the API from src/scrutatio/ yields a config with
        # no API key, and /match answers 503 openrouter_unconfigured on a machine
        # where the key is sitting right there in the file.
        from scrutatio.config import _env_file

        here = _env_file()
        monkeypatch.chdir(tmp_path)
        assert _env_file() == here
        assert Path(str(here)).is_absolute()

    def test_an_absolute_override_is_left_alone(self, clean_env: None) -> None:
        # DB_PATH=/data/scrutatio.duckdb is what the Dockerfile passes, and
        # anchoring it to a checkout that does not exist there would break the image.
        explicit = Path("/data/elsewhere.duckdb").resolve()
        assert _settings(db_path=explicit).db_path == explicit

    def test_unconfigured_by_default(self, clean_env: None) -> None:
        assert _settings().openrouter_configured is False


class TestSecretHandling:
    def test_key_is_not_exposed_in_repr(self, clean_env: None) -> None:
        s = _settings(openrouter_api_key=DUMMY_KEY)
        assert DUMMY_KEY not in repr(s)
        assert DUMMY_KEY not in str(s)

    def test_key_is_not_exposed_in_model_dump(self, clean_env: None) -> None:
        s = _settings(openrouter_api_key=DUMMY_KEY)
        assert DUMMY_KEY not in str(s.model_dump())

    def test_key_is_retrievable_deliberately(self, clean_env: None) -> None:
        s = _settings(openrouter_api_key=DUMMY_KEY)
        assert s.openrouter_api_key is not None
        assert s.openrouter_api_key.get_secret_value() == DUMMY_KEY


class TestEnvironmentOverride:
    def test_env_vars_are_read(self, clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", DUMMY_KEY)
        assert _settings().openrouter_configured is True

    def test_env_var_names_are_case_insensitive(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("openrouter_api_key", DUMMY_KEY)
        assert _settings().openrouter_api_key is not None

    def test_model_can_be_overridden_without_a_code_change(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The calibration gate compares candidates; swapping one must not require
        # an edit, and it changes the extraction signature on its own.
        monkeypatch.setenv("EXTRACTION_MODEL", "openai/gpt-5-nano")
        assert _settings().extraction_model == "openai/gpt-5-nano"


class TestChatCompletionsUrl:
    def test_builds_the_openrouter_endpoint(self, clean_env: None) -> None:
        assert _settings().chat_completions_url.endswith("/chat/completions")

    def test_tolerates_trailing_slash_on_base_url(self, clean_env: None) -> None:
        s = _settings(openrouter_base_url="https://openrouter.ai/api/v1/")
        assert "//chat/completions" not in s.chat_completions_url


class TestValidation:
    def test_blank_scope_filter_is_rejected(self, clean_env: None) -> None:
        with pytest.raises(ValidationError, match="bounds the entire corpus"):
            _settings(scope_filter="   ")

    def test_scope_filter_is_stripped(self, clean_env: None) -> None:
        assert _settings(scope_filter="  AREA[X]Y  ").scope_filter == "AREA[X]Y"

    @pytest.mark.parametrize("size", [0, -1, 1001])
    def test_page_size_bounds(self, clean_env: None, size: int) -> None:
        # CT.gov coerces anything above 1000 down to 1000; catch it locally instead.
        with pytest.raises(ValidationError):
            _settings(page_size=size)

    def test_negative_timeout_is_rejected(self, clean_env: None) -> None:
        with pytest.raises(ValidationError):
            _settings(request_timeout_seconds=0)

    def test_spend_ceiling_must_be_positive(self, clean_env: None) -> None:
        with pytest.raises(ValidationError):
            _settings(max_spend_usd=0)

    @pytest.mark.parametrize("workers", [0, -1, 65])
    def test_worker_count_is_bounded(self, clean_env: None, workers: int) -> None:
        # Concurrency against an undocumented rate limit is the failure mode that
        # cost a previous run; an accidental 200 must not be reachable by typo.
        with pytest.raises(ValidationError):
            _settings(extraction_workers=workers)


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()
