Homework for RS School Machine Learning course.
This project uses [Forest train dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction).

## Usage
This package allows train model for integer classification for the forest cover type (seven types). 
1. Clone this repository to your machine.
2. Download [Forest train](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset, save csv locally (default path is *data/heart.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.11).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Run train with the following command in order to train one model with KFold cross-validation:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as classifiers and hyperparameters (including max_depth, n_estimators for RandomForestClassifier and kernel, C, gamma for SVC)) in the CLI. 
To get a full list of them, use help:
```sh
poetry run train --help
```

6. Run model_selection with the following command in order to conduct GridSearch with nested cross-validation.
Parameters for Grid Search are placed in parameters.py file and can be added manually.
```sh
poetry run model_selection -d <path to csv with data> 
```
You can configure additional options in the CLI.  To get a full list of them, use help:
```sh
poetry run model_selection --help
```
7. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```

## Experiments 
### (Task 7)
Four metrics were used for evaluation: accuracy, f1, recall, precision. F1 was used for final choosing of best model.
Details about metrics realisation can be found in *train.py* and *model_selection.py* files.
KFold cross validation is used for training separate models (command *train*, file *train.py*).
Nested cross validation is used for automatic hyperparameter search (command *model_selection*, file *model_selection.py*).

### (Task 8)
1. Using command *train* five models were evaluated (DecisionTreeClassifier,
            RandomForestClassifier,
            AdaBoostClassifier,
            GradientBoostingClassifier,
            SVC) with default hyperparameters.
2. Two models (RandomForestClassifier, SVC) were used with at least three different sets of hyperparameters for each model. 
3. Two best models from previous step were used with at least tree different feature engineering techniques for each model:
   1. StandartScaler,
   2. QuantileTransformer (for first 10 columns in dataset),
   3. PolynomialFeatures (for first 10 columns in dataset). 
   Feature engineering techniques can be found in *pipeline.py* file.

Only part of experiments are provided on screenshot.![ml_flow_exper_screen](https://user-images.githubusercontent.com/89841675/166705198-52ac9cff-d6eb-4d91-9740-2ccb2f97d57b.png)
 The Best results (accuracy: 0.855, f1: 0.853, recall: 0.855, precision: 0.853) was obtained with following parameters:
```sh
poetry run train --estimator=RandomForestClassifier() --forest-param=gini 32 150 --use-scaler=True --use-uniform=True --use-poly=False
```

### (Task 9)
Command *model_selection* were developed for conducting automatic hyperparameter search for each model. Code is provided in *model_selection.py* file.
Command model_selection run Grid Search with parameters that are placed in *parameters.py* file and can be added manually.
Function runs Grid Search using nested cross validation. Only best model (or models, if metrics are equal) is added to MLFlow. 

New classification algorithms can be added to function if they are Sklearn classifiers. For using other algorithms they must be imported through
pipeline.py file (for example: *from sklearn.neighbors import KNeighborsClassifier*).
Please note, if you decide to rerun model_selection with parameters from parameters.py file, it will take quite a lot time.
Five models were tested with different parameters and different feature engineering techniques.

The best model always was RandomForestClassifier (possibly because I didn't find good hyperparameters for other models).
Part of experiments is provided on screenshot:
![ml_flow_grid_screen](https://user-images.githubusercontent.com/89841675/166950723-8445cce5-dd6e-4b2c-8fbd-4e67d2317b4b.png)
