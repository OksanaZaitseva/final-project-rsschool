import click
from click.testing import CliRunner
import pytest
import numpy as np
import pandas as pd
# from forest_ml.model_selection import model_selection
from forest_ml.data import get_dataset
from forest_ml.train import train

import os

@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()

def test_error_for_invalid_dataset() -> None:
    click.echo(os.listdir())
    """It fails when loaded dataset doesn't contain appropriate data."""
    try:
        features_train, features_val, target_train, target_val = get_dataset("tests/test_dataset.csv", 42, 0.2)
        col_names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
           'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
           'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
           'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
           'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
           'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
           'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
           'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
           'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
           'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
           'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
           'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
           'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
           'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
           'Soil_Type39', 'Soil_Type40']
        assert np.array_equal(features_train.columns, col_names)
    except ValueError:

        click.echo('Inappropriate Dataset')
        raise
    except KeyError:
        click.echo('Inappropriate Dataset')
        raise

def test_keep_dir(
        runner: CliRunner,
        tmp_path='tests/tem_dirs',

) -> None:
    """Filesystem isolation, and checking saved model for correctness"""
    # runner = CliRunner()
    df = pd.read_csv("tests/test_dataset.csv")
    with runner.isolated_filesystem(temp_dir=tmp_path) as td:
        df.to_csv('data.csv', index=False)


        result = runner.invoke(
            # train, ["--save-model-path", '/'.join([tmp_path, 'model.joblib']),
                    train, ["--save-model-path", 'model.joblib',
                            "--dataset-path", "data.csv",
                            "--cv-n", 2],
        )
        click.echo(result.output)
        assert result.exit_code == 0