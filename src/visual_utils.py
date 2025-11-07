import matplotlib.pyplot as plt


def plot_metrics(df_results, task_type):
    """
    Plot model performance metrics based on the task type.
    Parameters:
    - df_results: pd.DataFrame - DataFrame containing model performance metrics.
    - task_type: str - The type of task, either "Classification" or "Regression".
    Returns:
    - plt.Figure: A matplotlib Figure object containing the plot.
    """

    plt.figure(figsize=(8, 5))  # Set figure size

    if task_type == "Classification":
        plt.bar(df_results["model"], df_results["test_accuracy"], label="Accuracy")
        plt.bar(df_results["model"], df_results["f1"], alpha=0.6, label="F1 Score")
        plt.ylabel("Score")
    else:
        plt.bar(df_results["model"], df_results["r2"], label="R²")
        plt.bar(df_results["model"], df_results["mae"], alpha=0.6, label="MAE")
        plt.ylabel("Regression metric")

    plt.title("Model Comparison")
    plt.legend()
    plt.tight_layout()

    return plt.gcf()