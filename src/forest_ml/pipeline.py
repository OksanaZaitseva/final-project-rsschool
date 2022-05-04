from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer

from .classifier_switcher import ClfSwitcher

classifiers = {'DecisionTreeClassifier()': DecisionTreeClassifier(),
               'RandomForestClassifier()': RandomForestClassifier(),
               'AdaBoostClassifier()': AdaBoostClassifier(),
               'GradientBoostingClassifier()': GradientBoostingClassifier(),
               'SVC()': SVC()}

norm_col = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
            'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
            'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
            'Horizontal_Distance_To_Fire_Points']


def create_pipeline(
        use_scaler: bool, uniform: bool, poly: bool,
        estimator: str = 'DecisionTreeClassifier()', degree: int = 2, norm_col=norm_col
) -> Pipeline:
    preproc_norm = ColumnTransformer(
        [('quantile', QuantileTransformer(output_distribution='normal', random_state=random_state), norm_col)],
        remainder='passthrough')

    preproc_poly = ColumnTransformer(
        [('polynom', PolynomialFeatures(degree, interaction_only=True), norm_col)],
        remainder='passthrough')

    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))

    if uniform:
        pipeline_steps.append(("uniform", preproc_norm))

    if poly:
        pipeline_steps.append(("polynom", preproc_poly))

    pipeline_steps.append(("classifier", ClfSwitcher(classifiers[estimator])))
    return Pipeline(steps=pipeline_steps)

