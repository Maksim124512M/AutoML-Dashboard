import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, 
                             mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error)

from .data_utils import preprocess_data

def train_model(df, target, task_type):
    """
    Train machine learning models based on the specified task type (Classification or Regression).
     Returns a DataFrame with model performance metrics.
    Parameters:
    - df: pd.DataFrame - The input data containing features and target variable.
    - target: str - The name of the target variable column.
    - task_type: str - The type of task, either "Classification" or "Regression".
    Returns:
    - pd.DataFrame: A DataFrame containing model performance metrics.
    """
    
    X_train, X_test, y_train, y_test = preprocess_data(df=df, target_col=target)
    
    results = []
    best_models = {}
    if task_type == "Classification":
        models = {
            "LogReg": LogisticRegression(max_iter=5000),
            "SVM": SVC(),
            "RF": RandomForestClassifier(), 
        }

        grid_params = {
            "LogReg": {"C": [0.1, 1, 10]},
            "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
            "RF": {"n_estimators": [100, 200], "max_depth": [None, 10]}
        }

    if task_type == "Regression":
        models = {
            "LinReg": LinearRegression(),
            "SVR": SVR(),
            "RF": RandomForestRegressor(),
        }

        grid_params = {
            "LinReg": {},
            "SVR": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
            "RF": {"n_estimators": [100, 200], "max_depth": [None, 10]},
        }

    for name, model in models.items():
        if task_type == "Regression": 
            scoring = "r2"
        else:
            scoring = "accuracy"

        grid = GridSearchCV(model, grid_params[name], cv=5, scoring=scoring, verbose=0, error_score="raise")
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        base_metrics = {
            "model": name,
            "cv_score": grid.best_score_,
            "best_params": grid.best_params_,
        }

        if task_type == "Regression":
            mse = mean_squared_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            base_metrics.update({
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            })
        elif task_type == "Classification":
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
        
            base_metrics.update({
                "test_accuracy": accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            })

        results.append(base_metrics)        

        best_models[name] = best_model
    
    df_results = pd.DataFrame(results)

    if task_type == "Classification":
        df_results.sort_values("test_accuracy", ascending=False, inplace=True)
    else:
        df_results.sort_values("r2", ascending=False, inplace=True)

    df_results.reset_index(drop=True, inplace=True)

    return df_results