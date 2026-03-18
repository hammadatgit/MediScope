from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)


# -------- REGRESSION --------
def evaluate_regression(y_true, predictions):

    results = {}

    for name, pred in predictions.items():

        mse = mean_squared_error(y_true, pred)
        r2 = r2_score(y_true, pred)

        results[name] = {
            "MSE": round(mse, 2),
            "R2": round(r2, 4)
        }

    return results


# -------- CLASSIFICATION --------
def evaluate_classification(y_true, predictions, probas):

    results = {}

    for name in predictions:

        acc = accuracy_score(y_true, predictions[name])
        prec = precision_score(y_true, predictions[name])
        rec = recall_score(y_true, predictions[name])
        roc = roc_auc_score(y_true, probas[name])
        cm = confusion_matrix(y_true, predictions[name])

        results[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "ROC_AUC": round(roc, 4),
            "ConfusionMatrix": cm.tolist()
        }

    return results