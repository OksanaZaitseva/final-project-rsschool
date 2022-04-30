from pathlib import Path

import click
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

from .data import get_dataset
from .pipeline import create_pipeline

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int)

@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
)

@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)

def train(dataset_path: Path,
          save_model_path: Path,
          random_state: int,
          test_split_ratio: float,
          use_scaler: bool,

          ) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    pipeline = create_pipeline(use_scaler, random_state)
    pipeline.fit(features_train, target_train)

    predict = pipeline.predict(features_val)

    metric_results = []
    for metric in [accuracy_score, f1_score, recall_score, precision_score]:
        if metric == accuracy_score:
            metric_results.append(metric(target_val, predict))
        else:
            metric_results.append(metric(target_val, predict, average='micro'))

    click.echo(
        f"Accuracy: {metric_results[0]}, F1: {metric_results[1]}, Recall: {metric_results[2]}, Precision: {metric_results[3]}")