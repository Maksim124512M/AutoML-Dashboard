import pandas as pd

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(df: pd.DataFrame, target_col: str):
    """Preprocess the data by handling missing values, encoding categorical variables,
    and scaling features. Splits the data into training and testing sets."""

    df = df.copy()

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object"):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=target_col)  # Features
    y = df[target_col]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)   

    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance with SMOTE
    X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train_scaled, y_train)

    return X_train_resampled, X_test_scaled, y_train_resampled, y_test


    


