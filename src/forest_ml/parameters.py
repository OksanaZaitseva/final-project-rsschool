""" Dictionary 'parameters' contains parameters for GridSearch
    Dictionary is used by default. Content can be added for conducting  other experiments"""


import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)

parameters = [
    {
        "classifier__estimator": [DecisionTreeClassifier()],
        "classifier__estimator__criterion": ["gini", "entropy"],
        'classifier__estimator__max_depth': np.linspace(3, 15, 5, dtype=int),
    },
    {
        'classifier__estimator': [RandomForestClassifier()],
        'classifier__estimator__criterion': ['gini', 'entropy'],
        'classifier__estimator__max_depth': np.linspace(15, 50, 5, dtype=int),
        'classifier__estimator__n_estimators': np.linspace(50, 150, 5, dtype=int),
    },
    {
        'classifier__estimator': [AdaBoostClassifier()],
        'classifier__estimator__n_estimators': np.linspace(50, 150, 5, dtype=int),
    },
    {
        'classifier__estimator': [GradientBoostingClassifier()],
        'classifier__estimator__n_estimators': np.linspace(50, 150, 5, dtype=int),
        "classifier__estimator__max_depth":[3, 5, 8],
        'classifier__estimator__learning_rate': [0.8, 1.0]
    },
    {
        'classifier__estimator': [SVC()],
        'classifier__estimator__kernel': ['linear', 'rbf'],
        'classifier__estimator__C': [1, 10],
        'classifier__estimator__gamma': [0.001, 0.0001],
    },
]
