"""Module for Grid Search with nested cross validation:

   Inner loop calls scikit-learn’s GridSearchCV to achieve grid search of hyperparameter
        evaluated on the inner loop val set,
    Outer loop can calls KFold for generalization error.

    Only best models are added to MLFlow"""


from pathlib import Path

import click
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

from .data import get_dataset
from .pipeline import create_pipeline
from .classifier_switcher import ClfSwitcher
from .parameters import parameters

from sklearn.model_selection import GridSearchCV, KFold


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
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
    "--outer-splits",
    default=5,
    type=int)

@click.option(
    "--inner-splits",
    default=3,
    type=int)


def model_selection(dataset_path: Path,
          random_state: int,
          test_split_ratio: float,
          use_scaler: bool,
          use_uniform: bool,
          use_poly: bool,
          outer_splits: int,
          inner_splits: int
          ) -> None:
        features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
        metric_list = ('accuracy', 'f1_weighted',
                                    'recall_weighted', 'precision_weighted')
        params = {'use_scaler': use_scaler,
                  'use_uniform': use_uniform, 'use_poly': use_poly}
        outer_metrics = pd.DataFrame(columns=['Accuracy', 'F1', 'Recall', 'Precision'])
        best_params = []
        cv_outer = KFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
        for train_ix, test_ix in cv_outer.split(features_train):
            X_train, X_test = features_train.iloc[train_ix, :], features_train.iloc[test_ix, :]
            y_train, y_test = target_train.iloc[train_ix], target_train.iloc[test_ix]
            cv_inner = KFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
            pipeline = create_pipeline(use_scaler=True, use_uniform=False, use_poly=False, random_state=random_state)
            gscv = GridSearchCV(pipeline, parameters, cv=cv_inner,
                                scoring=metric_list, refit='f1_weighted')

            result = gscv.fit(X_train, y_train)
            best_model = result.best_estimator_
            yhat = best_model.predict(X_test)
            test_scores = {'Accuracy': accuracy_score(y_test, yhat),
                           'F1': f1_score(y_test, yhat, average='weighted'),
                           'Recall': recall_score(y_test, yhat, average='weighted'),
                           'Precision': precision_score(y_test, yhat, average='weighted')}
            outer_metrics = outer_metrics.append(test_scores, ignore_index=True)
            best_params.append(gscv.best_params_)

        params_best = pd.DataFrame(best_params)
        params_best.columns = [x.rsplit('__')[-1] for x in params_best.columns]
        params_best = params_best.merge(outer_metrics, left_index=True, right_index=True)
        num_params = params_best.shape[1] - 4
        params_best = params_best.groupby(params_best.columns[:num_params].to_list()).mean().reset_index()
        click.echo(params_best)

        for x in range(params_best.shape[0]):
            with mlflow.start_run():
                params_add = params_best.loc[x, params_best.columns[:num_params]].to_dict()
                params_add.update(params)
                mlflow.log_params(params_add)
                metr = params_best.loc[x, params_best.columns[num_params:len(params_best.columns)]].to_dict()
                mlflow.log_metrics(metr)




    