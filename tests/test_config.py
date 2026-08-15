"""Config tests, including the security property that matters most:
a token must never surface in a repr, log line, or traceback.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scrutatio.config import (
    DEFAULT_SCOPE_FILTER,
    EMBEDDING_DIMENSIONS,
    Settings,
    get_settings,
)

DUMMY_HOST = "https://dbc-00000000-0000.cloud.databricks.com"
DUMMY_TOKEN = "fake-token-for-tests-only"


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate from the developer's real .env and shell environment."""
    for var in (
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
        "CHAT_ENDPOINT",
        "EMBEDDING_ENDPOINT",
        "SCOPE_FILTER",
        "PAGE_SIZE",
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

    def test_default_chat_model_returns_plain_string_content(self, clean_env: None) -> None:
        # gpt-oss models return `content` as an array of typed parts, which
        # breaks standard OpenAI clients. The default must not be one of them.
        assert "gpt-oss" not in _settings().chat_endpoint

    def test_default_embedding_model_has_the_long_context_window(self, clean_env: None) -> None:
        # bge-large-en truncates at 512 tokens; eligibility texts average ~705.
        assert _settings().embedding_endpoint == "databricks-gte-large-en"

    def test_embedding_dimensions_match_measured_value(self) -> None:
        assert EMBEDDING_DIMENSIONS == 1024

    def test_recruiting_only_by_default(self, clean_env: None) -> None:
        assert _settings().recruiting_only is True

    def test_unconfigured_by_default(self, clean_env: None) -> None:
        assert _settings().databricks_configured is False


class TestSecretHandling:
    def test_token_is_not_exposed_in_repr(self, clean_env: None) -> None:
        s = _settings(databricks_host=DUMMY_HOST, databricks_token=DUMMY_TOKEN)
        assert DUMMY_TOKEN not in repr(s)
        assert DUMMY_TOKEN not in str(s)

    def test_token_is_not_exposed_in_model_dump(self, clean_env: None) -> None:
        s = _settings(databricks_host=DUMMY_HOST, databricks_token=DUMMY_TOKEN)
        assert DUMMY_TOKEN not in str(s.model_dump())

    def test_token_is_retrievable_deliberately(self, clean_env: None) -> None:
        s = _settings(databricks_token=DUMMY_TOKEN)
        assert s.databricks_token is not None
        assert s.databricks_token.get_secret_value() == DUMMY_TOKEN


class TestEnvironmentOverride:
    def test_env_vars_are_read(self, clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABRICKS_HOST", DUMMY_HOST)
        monkeypatch.setenv("DATABRICKS_TOKEN", DUMMY_TOKEN)
        s = _settings()
        assert s.databricks_configured is True

    def test_env_var_names_are_case_insensitive(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("databricks_host", DUMMY_HOST)
        assert _settings().databricks_host is not None


class TestServingEndpointUrl:
    def test_builds_invocation_url(self, clean_env: None) -> None:
        s = _settings(databricks_host=DUMMY_HOST)
        url = s.serving_endpoint_url("databricks-gte-large-en")
        assert url == f"{DUMMY_HOST}/serving-endpoints/databricks-gte-large-en/invocations"

    def test_tolerates_trailing_slash_on_host(self, clean_env: None) -> None:
        s = _settings(databricks_host=DUMMY_HOST + "/")
        assert "//serving-endpoints" not in s.serving_endpoint_url("x")

    def test_raises_without_host(self, clean_env: None) -> None:
        with pytest.raises(ValueError, match="databricks_host"):
            _settings().serving_endpoint_url("x")


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

    def test_unknown_chat_endpoint_is_rejected(self, clean_env: None) -> None:
        with pytest.raises(ValidationError):
            _settings(chat_endpoint="gpt-4")

    def test_negative_timeout_is_rejected(self, clean_env: None) -> None:
        with pytest.raises(ValidationError):
            _settings(request_timeout_seconds=0)


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()
