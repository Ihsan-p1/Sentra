"""
Unit tests for the destructive-test guard.

tests/test_schema_integration.py truncates the article tables. Pointed at the
development database it emptied a freshly embedded 154-article corpus in under a
second. The guard is what stops that happening again, so it is tested offline -
it has to hold without a database present.
"""
import pytest

from tests.test_schema_integration import (
    REQUIRED_DB_NAME_MARKERS,
    assert_disposable,
    target_database_url,
)


class TestAssertDisposable:
    @pytest.mark.parametrize("url", [
        "postgresql://u:p@localhost:5432/sentra_test",
        "postgresql://u:p@localhost:5432/sentra_ci",
        "postgresql://u:p@localhost:5432/ci_sentra",
        "postgresql://u:p@localhost:5432/scratch",
        "postgresql://u:p@localhost:5432/TEST_SENTRA",
    ])
    def test_disposable_names_are_allowed(self, url):
        assert_disposable(url)  # must not skip

    @pytest.mark.parametrize("url", [
        "postgresql://u:p@localhost:5432/Sentra1",
        "postgresql://u:p@localhost:5432/sentra",
        "postgresql://u:p@localhost:5432/production",
        "postgresql://u:p@localhost:5432/postgres",
    ])
    def test_real_looking_names_are_refused(self, url):
        with pytest.raises(BaseException) as exc:
            assert_disposable(url)
        assert "refusing to truncate" in str(exc.value)

    def test_the_actual_dev_database_is_refused(self):
        """The specific database that was wiped."""
        with pytest.raises(BaseException, match="refusing to truncate"):
            assert_disposable("postgresql://postgres:pass@localhost:5432/Sentra1")

    def test_query_parameters_do_not_smuggle_a_marker(self):
        """?sslmode=test must not make a production database look disposable."""
        with pytest.raises(BaseException, match="refusing to truncate"):
            assert_disposable("postgresql://u:p@localhost:5432/Sentra1?sslmode=test")

    def test_trailing_slash_is_tolerated(self):
        assert_disposable("postgresql://u:p@localhost:5432/sentra_ci/")

    def test_the_skip_message_says_how_to_proceed(self):
        with pytest.raises(BaseException) as exc:
            assert_disposable("postgresql://u:p@localhost:5432/Sentra1")
        message = str(exc.value)
        assert "SENTRA_TEST_DATABASE_URL" in message
        assert all(marker in message for marker in REQUIRED_DB_NAME_MARKERS)


class TestTargetDatabaseUrl:
    def test_override_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("SENTRA_TEST_DATABASE_URL", "postgresql://x/sentra_test")
        assert target_database_url() == "postgresql://x/sentra_test"

    def test_falls_back_to_settings(self, monkeypatch):
        monkeypatch.delenv("SENTRA_TEST_DATABASE_URL", raising=False)
        from config.settings import settings
        assert target_database_url() == settings.DATABASE_URL
