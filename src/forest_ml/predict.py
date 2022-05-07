import logging
from pathlib import Path
import joblib


import click
import numpy as np
import pandas as pd


# from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from .supp_functions import metrics_group

# from .data import get_dataset

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/test.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-m",
    "--model_path",
    default="data/model.joblib",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-submit-path",
    default="data/submission.csv",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)


# @click.option(
#     "--test-split-ratio",
#     default=0.2,
#     type=click.FloatRange(0, 1, min_open=True, max_open=True),
# )
#
# @click.option(
#     "--use-scaler",
#     default=True,
#     type=bool,
#     show_default=True,
# )
#
# @click.option(
#     "--use-uniform",
#     default=False,
#     type=bool,
#     show_default=True,
# )
#
# @click.option(
#     "--use-poly",
#     default=False,
#     type=bool,
#     show_default=True,
# )
#
# @click.option(
#     "--random_state",
#     default=42,
#     type=int)
#
# @click.option(
#     "--cv-n",
#     default=5,
#     type=click.IntRange(1, 21, min_open=True, max_open=True))

# @click.option(
#     "--estimator",
#     default='DecisionTreeClassifier()',
#     type=click.Choice(['DecisionTreeClassifier()',
#             'RandomForestClassifier()',
#             'AdaBoostClassifier()',
#             'GradientBoostingClassifier()',
#             'SVC()'], case_sensitive=True),
#     show_default=True
# )
#
# @click.option('--forest-param', nargs=2, type=click.Tuple([int, int]))
#
# @click.option('--svc-param', nargs=3, type=click.Tuple([str, int, float]))

def predict(model_path: Path,
          dataset_path: Path,
        save_submit_path: Path
          # random_state: int,
          # test_split_ratio: float,
          # use_scaler: bool,
          # use_uniform: bool,
          # use_poly: bool,
          # estimator: str,
          # forest_param: tuple,
          # svc_param: tuple,
          # cv_n: int
          ) -> None:

    # features_train, features_val, target_train, target_val = get_dataset(
    #     dataset_path,
    #     random_state,
    #     test_split_ratio,
    # )
    try:
        loaded_model = joblib.load(model_path)
    except FileNotFoundError:
         click.echo("Model doesn't exist. Please run 'poetry run train ...' at first")
         return()
    click.echo('Model loaded')
    dataset = pd.read_csv(dataset_path, index_col='Id')
    click.echo(f"Dataset shape: {dataset.shape}.")
    if 'Cover_Type' in dataset.columns:
        features = dataset.drop('Cover_Type', axis=1)
        target = dataset['Cover_Type']
        predicted = loaded_model.predict(features)
        test_metr = metrics_group(target, predicted)
        click.echo(test_metr)
    else:
        predicted = loaded_model.predict(dataset)
        pred = pd.DataFrame(predicted, columns=['Cover_Type'], index=dataset.index)

        pred.to_csv(save_submit_path)
        click.echo('Submission saved')