
Homework for RS School Machine Learning course.
This project uses [Forest train dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction).

## Usage
This package allows train model for integer classification for the forest cover type (seven types) and predict target feature on new dataset. 
1. Clone this repository to your machine.
2. Download [Forest train](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.11).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Package includes three commands for model selection, evaluation and prediction:
   5.1. Run train with the following command in order to train one model with KFold cross-validation:
    ```sh
    poetry run train -d <path to csv with data> -s <path to save trained model>
    ```
    You can configure additional options (such as classifiers and hyperparameters (including criterion, max_depth, n_estimators for RandomForestClassifier and kernel, C, gamma for SVC)) in the CLI. 
    To get a full list of them, use help:
    ```sh
    poetry run train --help
    ```

   5.2. Run model_selection with the following command in order to conduct GridSearch with nested cross-validation.
   Parameters for Grid Search are placed in parameters.py file and can be added manually. Function is for choosing model. If you need to save model, plead run command train with chosen parameters.
    ```sh
    poetry run model_selection -d <path to csv with data> 
    ```
    You can configure additional options in the CLI.  To get a full list of them, use help:
    ```sh
    poetry run model_selection --help
    ```
   5.3. Run predict with the following command in order to predict target values for new dataset using previously received during training model (train command).  If model wasn't previously received, you will get warning notification. You can use dataset with target column and receive metrics and saved dataset with prediction,
or use dataset without target value and receive .csv file according to kaggle submission. You can check this command with test.csv file from [Forest dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction).  Please, download the file previously and save it locally (default path is *data/test.csv* in repository's root).
    ```sh
    poetry run predict -d <path to csv with data> -m <path to model> -s <path to save received dataset>
    ```

6. Run MLflow UI to see the information about experiments you conducted with commands train and model_selection:
```sh
poetry run mlflow ui
```

## Development

The code in this repository was tested, formatted with black, and pass mypy typechecking before being commited to the repository.
Development was condacted on Windows 10. Some corrections was added to dev files in comperison with demo project files.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Now you can use developer instruments, e.g. pytest:
```
poetry run pytest
```
Code was formatted with [black](https://github.com/psf/black):
```
poetry run black src tests noxfile.py
```
Mypy:
```
poetry run mypy src tests noxfile.py
```
To run all sessions of testing and formatting in a single command, [nox](https://nox.thea.codes/en/stable/) was installed and used : 
```
python -m nox -r
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

### (Task 11)
All tests are in test folder. Type of tests:
    Four tests for error cases without using fake/sample data and filesystem isolation, as in the demo. 
    Three tests for a valid input case with test data, filesystem isolation, and checking saved model for correctness. Input datasets are in tests/tem_dir folder

### (Task 12)
Code is formatted with black and lint with flake8:
![black](https://user-images.githubusercontent.com/89841675/167271178-9acec3e3-e296-4d92-9fbc-c22705288ab8.png)

### (Task 13)
Code is type annotated with mypy:
![mypy](https://user-images.githubusercontent.com/89841675/167301459-bdd351eb-a9f7-46f5-8720-f456b56c3904.png)

### (Task 14)
To combine steps of testing and linting into a single command, nox was used:
![nox](https://user-images.githubusercontent.com/89841675/167301464-8c8f534f-e88f-4aab-bb02-1f5b462ad40d.png)
