import numpy as np
import pandas as pd
import click
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def parameters_check(parameters):
    res = {}
    for estim in parameters:
        params = [x.rsplit('__')[-1] for x in estim.keys() if x != 'classifier__estimator']
        check = [param not in estim['classifier__estimator'][0].get_params().keys() for param in params]
        res[estim['classifier__estimator'][0]] = [i for (i, v) in zip(params, check) if v]
        res = {x: res[x] for x in res.keys() if len(res[x]) > 0}
    if len(res) > 0:
        click.echo(f'Some values are not in parameters of estimators: {res}')
        return (True)
    else:
        click.echo("Parameters are checked")

def metrics_group(y_true, y_pred):
    test_scores = {'Accuracy': accuracy_score(y_true, y_pred),
                   'F1': f1_score(y_true, y_pred, average='weighted'),
                   'Recall': recall_score(y_true, y_pred, average='weighted'),
                   'Precision': precision_score(y_true, y_pred, average='weighted')}
    return test_scores