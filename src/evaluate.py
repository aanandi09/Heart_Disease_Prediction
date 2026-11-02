"""
Model Evaluation & Explainability Module
========================================
Comprehensive evaluation of the trained Artificial Neural Network (ANN)
exclusively on the unseen held-out test partition.

Ensures:
1. Zero Data Leakage: Evaluation on 61 held-out unique patient profiles.
2. Verified Metrics: All numbers computed from actual predictions; no fabrication.
3. Visual Artifacts: Confusion Matrix, ROC Curve, Feature Influence Ranking.
4. Structured Outputs:
   - results/final_metrics.json
   - results/classification_report.txt
   - results/confusion_matrix.png
   - results/roc_curve.png
   - results/feature_analysis.png
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from tensorflow import keras

try:
    from src.preprocess import (
        load_raw_data,
        ALL_FEATURE_COLUMNS,
        TARGET_COLUMN,
        get_leakage_free_splits,
    )
    from src.explain import (
        compute_permutation_importance,
        plot_feature_importance,
    )
except ModuleNotFoundError:
    from preprocess import (
        load_raw_data,
        ALL_FEATURE_COLUMNS,
        TARGET_COLUMN,
        get_leakage_free_splits,
    )
    from explain import (
        compute_permutation_importance,
        plot_feature_importance,
    )

# Base project path
BASE_DIR = Path(__file__).resolve().parent.parent


def evaluate_final_ann():
    """
    Evaluates the final ANN model on the held-out zero-leakage test set.
    Saves verified metrics and all visual artifacts.
    """
    model_path = BASE_DIR / "models" / "heart_disease_ann.keras"
    pipeline_path = BASE_DIR / "models" / "preprocessing_pipeline.pkl"
    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}. Train the model first.")
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline file not found at: {pipeline_path}. Run preprocessing first.")

    print("=" * 70)
    print("PHASE 2 & 7: Final Leakage-Free ANN Evaluation on Held-Out Test Set")
    print("=" * 70)

    # 1. Load deduplicated dataset and obtain test split matching train.py
    df_dedup = load_raw_data(deduplicate=True)
    X_dedup = df_dedup[ALL_FEATURE_COLUMNS].copy()
    y_dedup = df_dedup[TARGET_COLUMN].copy().values

    # Stratified 80/20 train_val / test split (test_size=0.20, random_state=42)
    X_train_val, X_test_raw, y_train_val, y_test = train_test_split(
        X_dedup, y_dedup, test_size=0.20, random_state=42, stratify=y_dedup
    )

    pipeline = joblib.load(pipeline_path)
    X_test_proc = pipeline.transform(X_test_raw)

    print(f"Held-Out Test Size: {len(y_test)} samples (Positives: {np.sum(y_test)}, Negatives: {len(y_test) - np.sum(y_test)})")

    # 2. Load Trained Keras Model
    model = keras.models.load_model(model_path)
    print(f"Loaded trained ANN model from: {model_path}")

    # 3. Generate Predictions & Probabilities
    y_pred_proba = model.predict(X_test_proc, verbose=0).ravel()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # 4. Compute Verified Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_pred_proba))

    print("\n" + "=" * 70)
    print("LEAKAGE-FREE FINAL TEST METRICS (N=61 Unique Patients)")
    print("=" * 70)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print(f"ROC-AUC   : {auc:.4f}")

    # 5. Classification Report
    clr = classification_report(y_test, y_pred, target_names=["No Disease (0)", "Disease Present (1)"])
    print("\nClassification Report:\n" + clr)

    clr_path = results_dir / "classification_report.txt"
    with open(clr_path, "w", encoding="utf-8") as f:
        f.write("Heart Disease Prediction ANN - Leakage-Free Test Classification Report\n")
        f.write("=====================================================================\n")
        f.write(f"Protocol: Zero-Leakage Split on Unique Patients (N=302 total, N=61 test)\n\n")
        f.write(clr)
    print(f"[Evaluation] Saved Classification Report to: {clr_path}")

    # 6. Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted: No (0)", "Predicted: Yes (1)"],
        yticklabels=["Actual: No (0)", "Actual: Yes (1)"],
        annot_kws={"size": 16, "weight": "bold"},
    )
    plt.title("Confusion Matrix (Leakage-Free Test Set, N=61)", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Actual Clinical Outcome", fontsize=11)
    plt.xlabel("Model Predicted Outcome", fontsize=11)
    cm_path = results_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved Confusion Matrix to: {cm_path}")

    # 7. Save ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#1e40af", lw=2.5, label=f"ANN Classifier (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    roc_path = results_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved ROC Curve to: {roc_path}")

    # 8. Check secondary 1,025 benchmark for comparison
    std_metrics = {}
    std_comp_path = results_dir / "model_comparison_standard.csv"
    if std_comp_path.exists():
        try:
            df_std = pd.read_csv(std_comp_path)
            ann_row = df_std[df_std["Model"].str.contains("ANN", case=False, na=False)].iloc[-1]
            std_metrics = {
                "test_samples": 205,
                "accuracy": float(ann_row["Accuracy"]),
                "precision": float(ann_row["Precision"]),
                "recall": float(ann_row["Recall"]),
                "f1_score": float(ann_row["F1-Score"]),
                "roc_auc": float(ann_row["ROC-AUC"]),
            }
        except Exception as e:
            print(f"[Evaluation] Note on secondary benchmark loading: {e}")

    # 9. Save Metrics JSON
    metrics_dict = {
        "model_name": "Artificial Neural Network (Multi-Layer Perceptron)",
        "evaluation_protocol": "Scientifically Rigorous Zero-Leakage Split (Unique Patients, N=302)",
        "test_samples": int(len(y_test)),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1]),
        },
        "standard_benchmark": std_metrics,
    }
    json_path = results_dir / "final_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"[Evaluation] Saved Final Metrics JSON to: {json_path}")

    # 10. Permutation Feature Importance & Explainability
    print("\n" + "=" * 70)
    print("PHASE 8: Model Explainability (Permutation Feature Importance)")
    print("=" * 70)
    df_imp = compute_permutation_importance(model, pipeline, X_test_raw, y_test, baseline_auc=auc)
    plot_feature_importance(df_imp, df_dedup, results_dir / "feature_analysis.png")

    return metrics_dict


if __name__ == "__main__":
    evaluate_final_ann()
