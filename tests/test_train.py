from click.testing import CliRunner
import pytest

from forest_ml.train import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_test_split_ratio(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        ["--test-split-ratio", '42'],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--test-split-ratio'" in result.output


def test_error_for_invalid_cross_val_split(runner: CliRunner) -> None:
    """It fails when cross_val split  is 0."""
    result = runner.invoke(
        train,
        ["--cv-n", '0'],
    )
    assert result.exit_code == 2
    assert " Invalid value for '--cv-n'" in result.output
