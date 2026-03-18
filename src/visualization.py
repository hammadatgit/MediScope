import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import numpy as np
import os

# Create structured folders
BASE = "outputs/plots"

REG_PATH = f"{BASE}/regression"
CLS_PATH = f"{BASE}/classification"
CLUSTER_PATH = f"{BASE}/clustering"
PCA_PATH = f"{BASE}/pca"

os.makedirs(REG_PATH, exist_ok=True)
os.makedirs(CLS_PATH, exist_ok=True)
os.makedirs(CLUSTER_PATH, exist_ok=True)
os.makedirs(PCA_PATH, exist_ok=True)


# -------- REGRESSION --------
def plot_regression_coefficients(models):

    for name, model in models.items():

        if hasattr(model, "coef_"):

            coef = model.coef_

            plt.figure()
            plt.bar(range(len(coef)), coef)
            plt.title(f"{name} Coefficients")
            plt.xlabel("Features")
            plt.ylabel("Impact")

            plt.savefig(f"{REG_PATH}/{name}_coefficients.png")
            plt.close()


# -------- CONFUSION MATRIX --------
def plot_confusion_matrix(y_true, predictions):

    for name, pred in predictions.items():

        disp = ConfusionMatrixDisplay.from_predictions(y_true, pred)

        plt.title(f"{name} Confusion Matrix")
        plt.savefig(f"{CLS_PATH}/cm_{name}.png")
        plt.close()


# -------- ROC --------
def plot_roc_curve(y_true, probas):

    for name, proba in probas.items():

        RocCurveDisplay.from_predictions(y_true, proba)

        plt.title(f"{name} ROC Curve")
        plt.savefig(f"{CLS_PATH}/roc_{name}.png")
        plt.close()


# -------- CLUSTERS --------
def plot_clusters(X, labels, name):

    plt.figure()

    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.title(f"{name} Clustering")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.savefig(f"{CLUSTER_PATH}/{name}_clusters.png")
    plt.close()


# -------- PCA --------
def plot_pca(X_pca, labels):

    plt.figure()

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.title("PCA Visualization")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.savefig(f"{PCA_PATH}/pca.png")
    plt.close()