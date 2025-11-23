import pandas as pd
import numpy as np

from src.data_utils import preprocess_data

def test_preprocess_data_returns_splits():
    df = pd.read_csv("tests/Iris.csv")
    target = 'Species'

    X_train, X_test, y_train, y_test = preprocess_data(df, target)
    
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert y_train.isin([0, 1, 2]).all()


def test_preprocess_data_removes_nan():
    df = pd.DataFrame({
        'feature1': [1, None, 3],
        'feature2': [4, 5, None],
        'target': [0, 1, 0],
    })

    X_train, X_test, y_train, y_test = preprocess_data(df, target_col='target')
    assert not np.isnan(X_train).any().any()
    assert not np.isnan(X_train).any().any()