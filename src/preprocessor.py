import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def preprocess_data(df, config):

    target_cls = config["features"]["target_classification"]
    target_reg = config["features"]["target_regression"]

    # Separate features and targets
    X = df.drop([target_cls, target_reg], axis=1)
    y_cls = df[target_cls]
    y_reg = df[target_reg]

    # Identify column types
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Train-test split
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X, y_cls, y_reg,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"]
    )

    # Fit & transform
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return (
        X_train, X_test,
        y_cls_train, y_cls_test,
        y_reg_train, y_reg_test,
        preprocessor
    )