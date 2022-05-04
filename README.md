Homework for RS School Machine Learning course.
This project uses [Forest train dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction).

## Usage
This package allows train model for integer classification for the forest cover type (seven types). 
1. Clone this repository to your machine.
2. Download [Heart Disease](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) dataset, save csv locally (default path is *data/heart.csv* in repository's root).
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