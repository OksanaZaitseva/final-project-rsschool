import logging
from pathlib import Path
import joblib

import click
import numpy as np
import pandas as pd

from .supp_functions import metrics_group


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/test.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-m",
    "--model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-submit-path",
    default="data/submission.csv",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
def predict(model_path: Path, dataset_path: Path, save_submit_path: Path) -> None:
    try:
        loaded_model = joblib.load(model_path)
    except FileNotFoundError:
        click.echo("Model doesn't exist. Please run 'poetry run train ...' at first")
        return
    click.echo("Model loaded")
    dataset = pd.read_csv(dataset_path, index_col="Id")
    click.echo(f"Dataset shape: {dataset.shape}.")
    if "Cover_Type" in dataset.columns:
        features = dataset.drop("Cover_Type", axis=1)
        target = dataset["Cover_Type"]
        predicted = loaded_model.predict(features)
        test_metr = metrics_group(target, predicted)
        click.echo(test_metr)
        pred = pd.DataFrame(predicted, columns=["Cover_Type"], index=dataset.index)
        pred.to_csv(save_submit_path)
    else:
        predicted = loaded_model.predict(dataset)
        pred = pd.DataFrame(predicted, columns=["Cover_Type"], index=dataset.index)
        pred.to_csv(save_submit_path)
        click.echo("Submission saved")
