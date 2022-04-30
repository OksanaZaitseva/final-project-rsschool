from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .classifier_switcher import ClfSwitcher

def create_pipeline(
    use_scaler: bool, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        ("classifier", ClfSwitcher()),
        )
    return Pipeline(steps=pipeline_steps)
