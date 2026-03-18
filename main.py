import argparse
import os
import joblib

from src.logger import setup_logger
from src.loader import load_config, load_data, validate_data
from src.preprocessor import preprocess_data

from src.regression import train_regression_models, predict_regression
from src.classification import train_classification_models, predict_classification, predict_proba
from src.evaluation import evaluate_regression, evaluate_classification

from src.clustering import perform_kmeans
from src.dimensionality import apply_pca

from src.visualization import *
from src.report import generate_report, generate_patient_insight


# ------------------ Helper: Load Models ------------------
def load_models():
    """
    Load trained regression and classification models from outputs/models
    """
    reg_models = {}
    cls_models = {}

    model_path = "outputs/models"
    if not os.path.exists(model_path):
        print("⚠️ Models not found. Run with --train first")
        return reg_models, cls_models

    for file in os.listdir(model_path):
        if file.startswith("reg_") and file.endswith(".pkl"):
            reg_models[file[4:-4]] = joblib.load(os.path.join(model_path, file))
        elif file.startswith("cls_") and file.endswith(".pkl"):
            cls_models[file[4:-4]] = joblib.load(os.path.join(model_path, file))

    return reg_models, cls_models


# ------------------ Main Pipeline ------------------
def main():

    parser = argparse.ArgumentParser(description="MediScope ML Pipeline")
    parser.add_argument("--train", action="store_true", help="Train models and save")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models on test set")
    parser.add_argument("--plots", action="store_true", help="Generate plots")
    parser.add_argument("--report", action="store_true", help="Generate report and patient insight")

    args = parser.parse_args()

    setup_logger()

    # ---------------- Load & Preprocess ----------------
    config = load_config()
    df = load_data(config)
    validate_data(df)

    print("\n🔹 MediScope Pipeline Started")

    (
        X_train, X_test,
        y_cls_train, y_cls_test,
        y_reg_train, y_reg_test,
        preprocessor
    ) = preprocess_data(df, config)

    os.makedirs("outputs/models", exist_ok=True)

    # ---------------- TRAIN ----------------
    if args.train:
        print("\n⚙️ Training Models...")

        reg_models = train_regression_models(X_train, y_reg_train)
        cls_models = train_classification_models(X_train, y_cls_train)

        # Save models
        for name, model in reg_models.items():
            joblib.dump(model, f"outputs/models/reg_{name}.pkl")
        for name, model in cls_models.items():
            joblib.dump(model, f"outputs/models/cls_{name}.pkl")

        print("✅ Models trained & saved")

    # ---------------- EVALUATE ----------------
    if args.evaluate:
        print("\n📊 Evaluating Models...")

        reg_models, cls_models = load_models()
        if not reg_models or not cls_models:
            return

        reg_preds = predict_regression(reg_models, X_test)
        reg_results = evaluate_regression(y_reg_test, reg_preds)

        cls_preds = predict_classification(cls_models, X_test)
        cls_probas = predict_proba(cls_models, X_test)
        cls_results = evaluate_classification(y_cls_test, cls_preds, cls_probas)

        print("\n📊 Regression:")
        for k, v in reg_results.items():
            print(k, v)

        print("\n📊 Classification:")
        for k, v in cls_results.items():
            print(k, v)

    # ---------------- PLOTS ----------------
    if args.plots:
        print("\n📈 Generating Plots...")

        reg_models, cls_models = load_models()
        if not reg_models or not cls_models:
            return

        cls_preds = predict_classification(cls_models, X_test)
        cls_probas = predict_proba(cls_models, X_test)

        kmeans_labels = perform_kmeans(X_test)
        X_pca = apply_pca(X_test)

        plot_regression_coefficients(reg_models)
        plot_confusion_matrix(y_cls_test, cls_preds)
        plot_roc_curve(y_cls_test, cls_probas)
        plot_clusters(X_pca, kmeans_labels, "kmeans")
        plot_pca(X_pca, kmeans_labels)

        print("✅ Plots saved in outputs/plots")

    # ---------------- REPORT ----------------
    if args.report:
        print("\n🧾 Generating Report...")

        reg_models, cls_models = load_models()
        if not reg_models or not cls_models:
            return

        reg_preds = predict_regression(reg_models, X_test)
        reg_results = evaluate_regression(y_reg_test, reg_preds)

        cls_preds = predict_classification(cls_models, X_test)
        cls_probas = predict_proba(cls_models, X_test)
        cls_results = evaluate_classification(y_cls_test, cls_preds, cls_probas)

        generate_report(reg_results, cls_results)

        # 🔥 Patient-level insight
        generate_patient_insight(df.iloc[0])

    print("\n✅ Pipeline Completed")


if __name__ == "__main__":
    main()