from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# from .classifier_switcher import ClfSwitcher

classifiers={'DecisionTreeClassifier()': DecisionTreeClassifier(),
            'RandomForestClassifier()': RandomForestClassifier(),
            'AdaBoostClassifier()': AdaBoostClassifier(),
            'KNeighborsClassifier()': KNeighborsClassifier(),
            'SVC()': SVC()}


# try:
#     function=dispatcher[w]
# except KeyError:
#     raise ValueError('invalid input')

def create_pipeline(
    use_scaler: bool, estimator: str
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(("classifier", classifiers[estimator]))
    return Pipeline(steps=pipeline_steps)
