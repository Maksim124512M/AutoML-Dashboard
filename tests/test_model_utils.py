import pandas as pd

from src.model_utils import train_model

def test_train_model():
    df = pd.read_csv("tests/Iris.csv")
    target = "Species"
    task_type = "Classification"

    df_results = train_model(df=df, target=target, task_type=task_type)

    assert type(df_results) is pd.DataFrame

