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
from .supp_functions import metrics_group

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
    "--use-uniform",
    default=False,
    type=bool,
    show_default=True,
)

@click.option(
    "--use-poly",
    default=False,
    type=bool,
    show_default=True,
)

@click.option(
    "--random_state",
    default=42,
    type=int)

@click.option(
    "--cv-n",
    default=5,
    type=click.IntRange(1, 21, min_open=True, max_open=True))

@click.option(
    "--estimator",
    default='DecisionTreeClassifier()',
    type=click.Choice(['DecisionTreeClassifier()',
            'RandomForestClassifier()',
            'AdaBoostClassifier()',
            'GradientBoostingClassifier()',
            'SVC()'], case_sensitive=True),
    show_default=True
)

@click.option('--forest-param', nargs=3, type=click.Tuple([str, int, int]))

@click.option('--svc-param', nargs=3, type=click.Tuple([str, int, float]))

def train(dataset_path: Path,
          save_model_path: Path,
          random_state: int,
          test_split_ratio: float,
          use_scaler: bool,
          use_uniform: bool, 
          use_poly: bool,
          estimator: str,
          forest_param: tuple,
          svc_param: tuple,
          cv_n: int
          ) -> None:

    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )

    with mlflow.start_run():
        pipeline = create_pipeline(use_scaler, use_uniform, use_poly, random_state, estimator)
        params = {"estimator": estimator, 'use_scaler': use_scaler,
            'use_uniform': use_uniform, 'use_poly': use_poly}
        pipeline.set_params(classifier__estimator__random_state=random_state)

        if estimator == 'RandomForestClassifier()':
            params['criterion'], params['max_depth'], params['n_estimators'] = forest_param
            pipeline.set_params(classifier__estimator__criterion=params['criterion'],
                            classifier__estimator__max_depth=params['max_depth'],
                                classifier__estimator__n_estimators=params['n_estimators'])
        if estimator == 'SVC()':
            params['kernel'], params['C'], params['gamma']  = svc_param

            pipeline.set_params(classifier__estimator__kernel=params['kernel'],
                                classifier__estimator__C=params['C'],
                                classifier__estimator__gamma=params['gamma'])

        scores = cross_validate(pipeline, features_train, target_train,
                       scoring=('accuracy', 'f1_weighted',
                                'recall_weighted', 'precision_weighted'),
                       return_estimator=True, cv=cv_n)

        mlflow.log_params(params)
        click.echo(params)
        metr = {'Accuracy': np.mean(scores['test_accuracy']),
                   'F1': np.mean(scores['test_f1_weighted']),
                   'Recall': np.mean(scores['test_recall_weighted']),
                   'Precision': np.mean(scores['test_precision_weighted'])}
        mlflow.log_metrics(metr)
        click.echo(metr)
        pipeline.fit(features_train, target_train)
        test_pred = pipeline.predict(features_val)
        test_metr = metrics_group(target_val, test_pred)

        click.echo(test_metr)
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")