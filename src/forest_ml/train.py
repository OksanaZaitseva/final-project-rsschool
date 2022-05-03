from pathlib import Path
from joblib import dump

import click
import numpy as np
# import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from .data import get_dataset
from .pipeline import create_pipeline
from .classifier_switcher import ClfSwitcher

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

@click.option(
    "--random_state",
    default=42,
    type=int)

@click.option(
    "--estimator",
    default='DecisionTreeClassifier()',
    type=click.Choice(['DecisionTreeClassifier()',
            'RandomForestClassifier()',
            'AdaBoostClassifier()',
            'KNeighborsClassifier()',
            'SVC()'], case_sensitive=True),
    show_default=True
)


# @click.option('--dec_tree_', nargs=2, type=click.Tuple([str, int]))

def train(dataset_path: Path,
          save_model_path: Path,
          random_state: int,
          test_split_ratio: float,
          use_scaler: bool,
          estimator: str,
          
          ) -> None:

    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )

    with mlflow.start_run():
        pipeline = create_pipeline(use_scaler, estimator)
        pipeline.set_params(classifier__random_state=random_state)
        scores = cross_validate(pipeline, features_train, target_train,
                       scoring=('accuracy', 'f1_weighted',
                                'recall_weighted', 'precision_weighted'),
                       return_train_score=True
                       )
        click.echo(scores)
        mlflow.log_param("estimator", estimator)
        metrics = {'Accuracy': np.mean(scores['test_accuracy']),
                   'F1': np.mean(scores['test_f1_weighted']),
                   'Recall': np.mean(scores['test_recall_weighted']),
                   'Precision': np.mean(scores['test_precision_weighted']),}
        mlflow.log_metrics(metrics)
        click.echo(metrics)
        #
        # pipeline.fit(features_train, target_train, classifier__random_state=random_state)
        # predict = pipeline.predict(features_val)
        # metrics = {'Accuracy':accuracy_score(target_val, predict),
        #            'F1': f1_score(target_val, predict, average='micro'),
        #            'Recall': recall_score(target_val, predict, average='micro'),
        #            'Precision': precision_score(target_val, predict, average='micro')}
        # mlflow.log_param("estimator", estimator)
        # mlflow.log_metrics(metrics)
        # # mlflow.sklearn.log_model(pipeline, 'model')
        # click.echo(metrics)
        # dump(pipeline, save_model_path)
        # click.echo(f"Model is saved to {save_model_path}.")