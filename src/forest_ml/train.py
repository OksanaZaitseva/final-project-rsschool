from pathlib import Path

import click
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--random-state", default=42, type=int)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
)
def train(dataset_path: Path, random_state: int, test_split_ratio: float) -> None:
    dataset = pd.read_csv(dataset_path, index_col='Id')
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop('Cover_Type', axis=1)
    target = dataset['Cover_Type']

    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_split_ratio, random_state=random_state
    )
    classifier = RandomForestClassifier(random_state=random_state).fit(
        features_train, target_train
    )
    predict = classifier.predict(features_val)

    metric_results = []
    for metric in [accuracy_score, f1_score, recall_score, precision_score]:
        if metric == accuracy_score:
            metric_results.append(metric(target_val, predict))
        else:
            metric_results.append(metric(target_val, predict, average='micro'))

    click.echo(
        f"Accuracy: {metric_results[0]}, F1: {metric_results[1]}, Recall: {metric_results[2]}, Precision: {metric_results[3]}")