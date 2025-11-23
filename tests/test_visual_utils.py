import pandas as pd

from src.visual_utils import plot_metrics

from matplotlib.figure import Figure

from src.model_utils import train_model

def test_plot_metrics():
    df = pd.read_csv("tests/Iris.csv")
    target = "Species"
    task_type = "Classification"

    df_results = train_model(df=df, target=target, task_type=task_type)

    assert type(plot_metrics(df_results=df_results, task_type=task_type)) is Figure

