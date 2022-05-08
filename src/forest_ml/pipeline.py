import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    QuantileTransformer,
    PolynomialFeatures,
)
from sklearn.compose import ColumnTransformer

# from .classifier_switcher import ClfSwitcher

classifiers = {
    "DecisionTreeClassifier()": DecisionTreeClassifier(),
    "RandomForestClassifier()": RandomForestClassifier(),
    "AdaBoostClassifier()": AdaBoostClassifier(),
    "GradientBoostingClassifier()": GradientBoostingClassifier(),
    "SVC()": SVC(),
}


def create_pipeline(
    use_scaler: bool,
    use_uniform: bool,
    use_poly: bool,
    random_state: int,
    estimator: str = "DecisionTreeClassifier()",
    degree: int = 2,
) -> Pipeline:
    preproc_norm = ColumnTransformer(
        [
            (
                "quantile",
                QuantileTransformer(
                    output_distribution="normal", random_state=random_state
                ),
                np.linspace(0, 9, 9, dtype=int),
            )
        ],
        remainder="passthrough",
    )

    preproc_poly = ColumnTransformer(
        [
            (
                "polynom",
                PolynomialFeatures(degree, interaction_only=True),
                np.linspace(0, 9, 9, dtype=int),
            )
        ],
        remainder="passthrough",
    )

    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))

    if use_uniform:
        pipeline_steps.append(("uniform", preproc_norm))

    if use_poly:
        pipeline_steps.append(("polynom", preproc_poly))

    pipeline_steps.append(("classifier", classifiers[estimator]))

    return Pipeline(steps=pipeline_steps)
