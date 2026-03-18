from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def train_classification_models(X_train, y_train):

    models = {
        "Logistic": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(max_depth=5),
        "SVM": SVC(probability=True)
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def predict_classification(models, X_test):

    predictions = {}

    for name, model in models.items():
        predictions[name] = model.predict(X_test)

    return predictions


def predict_proba(models, X_test):

    probas = {}

    for name, model in models.items():
        probas[name] = model.predict_proba(X_test)[:, 1]

    return probas