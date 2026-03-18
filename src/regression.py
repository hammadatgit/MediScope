from sklearn.linear_model import LinearRegression, Ridge, Lasso


def train_regression_models(X_train, y_train):

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1)
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def predict_regression(models, X_test):

    predictions = {}

    for name, model in models.items():
        predictions[name] = model.predict(X_test)

    return predictions