import click
from click.testing import CliRunner
import pytest
from forest_ml.model_selection import model_selection
from forest_ml.parameters import parameters


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_test_split_ratio(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        model_selection,
        ["--test-split-ratio", "42"],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--test-split-ratio'" in result.output


def test_error_for_invalid_cross_val_split(runner: CliRunner) -> None:
    """It fails when cross_val split is 0."""
    result = runner.invoke(
        model_selection, ["--outer-splits", "0", "--inner-splits", "0"]
    )
    assert result.exit_code == 2
    assert "Invalid value for '--outer-splits'" in result.output
