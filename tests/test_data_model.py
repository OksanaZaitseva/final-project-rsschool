import click
from click.testing import CliRunner
import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from forest_ml.data import get_dataset
from forest_ml.train import train
from forest_ml.predict import predict


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_dataset() -> None:
    """It fails when loaded dataset doesn't contain appropriate data."""
    try:
        features_train, features_val, target_train, target_val = get_dataset(
            Path("tests/tem_dirs/test_dataset.csv"), 42, 0.2
        )
        col_names = [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
            "Wilderness_Area1",
            "Wilderness_Area2",
            "Wilderness_Area3",
            "Wilderness_Area4",
            "Soil_Type1",
            "Soil_Type2",
            "Soil_Type3",
            "Soil_Type4",
            "Soil_Type5",
            "Soil_Type6",
            "Soil_Type7",
            "Soil_Type8",
            "Soil_Type9",
            "Soil_Type10",
            "Soil_Type11",
            "Soil_Type12",
            "Soil_Type13",
            "Soil_Type14",
            "Soil_Type15",
            "Soil_Type16",
            "Soil_Type17",
            "Soil_Type18",
            "Soil_Type19",
            "Soil_Type20",
            "Soil_Type21",
            "Soil_Type22",
            "Soil_Type23",
            "Soil_Type24",
            "Soil_Type25",
            "Soil_Type26",
            "Soil_Type27",
            "Soil_Type28",
            "Soil_Type29",
            "Soil_Type30",
            "Soil_Type31",
            "Soil_Type32",
            "Soil_Type33",
            "Soil_Type34",
            "Soil_Type35",
            "Soil_Type36",
            "Soil_Type37",
            "Soil_Type38",
            "Soil_Type39",
            "Soil_Type40",
        ]
        assert np.array_equal(features_train.columns, col_names)
    except ValueError:
        click.echo("Inappropriate Dataset")
        raise
    except KeyError:
        click.echo("Inappropriate Dataset")
        raise


def test_keep_dir(
    runner: CliRunner,
    tmp_path="tests/tem_dirs",
) -> None:

    """Filesystem isolation, and checking saved model for correctness"""

    train_df = pd.read_csv("tests/tem_dirs/test_dataset.csv")
    test_df = pd.read_csv("tests/tem_dirs/test_predict.csv")
    test_kaggle = pd.read_csv("tests/tem_dirs/test_predict_kaggle.csv")
    with runner.isolated_filesystem(temp_dir=tmp_path):
        train_df.to_csv("train_df.csv", index=False)
        test_df.to_csv("test_df.csv", index=False)
        test_kaggle.to_csv("test_kaggle.csv", index=False)
        result_train = runner.invoke(
            train,
            [
                "--save-model-path",
                "model.joblib",
                "--dataset-path",
                "train_df.csv",
                "--cv-n",
                "2",
            ],
        )
        click.echo(result_train.output)
        assert result_train.exit_code == 0
        loaded_model = joblib.load("model.joblib")

        result_predict_test = runner.invoke(
            predict,
            [
                "--model-path",
                "model.joblib",
                "--dataset-path",
                "test_df.csv",
                "--save-submit-path",
                "submit.csv",
            ],
        )

        features = test_df.set_index("Id").drop("Cover_Type", axis=1)
        target = test_df["Cover_Type"]
        predicted = loaded_model.predict(features)
        np.unique(predicted)

        assert len(predicted) == len(target)
        assert set(predicted).issubset(np.linspace(1, 7, 7, dtype=int))
        assert result_predict_test.exit_code == 0

        result_predict_kaggle = runner.invoke(
            predict,
            [
                "--model-path",
                "model.joblib",
                "--dataset-path",
                "test_kaggle.csv",
                "--save-submit-path",
                "submit.csv",
            ],
        )
        click.echo("Predict kaggle function")
        click.echo(result_predict_kaggle.output)
        sub = pd.read_csv("submit.csv")

        assert len(sub) == len(test_kaggle)
        assert set(sub["Cover_Type"]).issubset(np.linspace(1, 7, 7, dtype=int))
        assert result_predict_test.exit_code == 0
