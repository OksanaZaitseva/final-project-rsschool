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
        "classifier": [DecisionTreeClassifier()],
        "classifier__criterion": ["gini", "entropy"],
        # 'classifier__max_depth': np.linspace(3, 15, 5, dtype=int),
    },
    {
        "classifier": [RandomForestClassifier()],
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": np.linspace(15, 50, 5, dtype=int),
        "classifier__n_estimators": np.linspace(50, 150, 5, dtype=int),
    },
    {
        "classifier": [AdaBoostClassifier()],
        "classifier__n_estimators": np.linspace(50, 150, 5, dtype=int),
    },
    {
        "classifier": [GradientBoostingClassifier()],
        "classifier__n_estimators": np.linspace(50, 150, 5, dtype=int),
        "classifier__max_depth": [3, 5, 8],
        "classifier__learning_rate": [0.8, 1.0],
    },
    {
        "classifier": [SVC()],
        "classifier__kernel": ["linear", "rbf"],
        "classifier__C": [1, 10],
        "classifier__gamma": [0.001, 0.0001],
    },
]

# parameters = [{
#         'classifier': [DecisionTreeClassifier()],
#         'classifier__criterion': ['gini', 'entropy'],
#         'classifier': [SVC()],
#         'classifier__C': [1, 10],
#
#     }]
