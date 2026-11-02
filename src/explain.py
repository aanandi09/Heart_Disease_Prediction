"""
Explainable AI (XAI) Module: Permutation Feature Influence
===========================================================
Computes and visualizes permutation feature importance for the
Artificial Neural Network (Multi-Layer Perceptron) on tabular cardiac data.

NOTE: Feature influence reflects how strongly the neural network relies
on an input feature to generate predictions. It indicates statistical
dependency and NOT medical or biological causation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

try:
    from src.preprocess import (
        load_raw_data,
        ALL_FEATURE_COLUMNS,
        TARGET_COLUMN,
    )
except ModuleNotFoundError:
    from preprocess import (
        load_raw_data,
        ALL_FEATURE_COLUMNS,
        TARGET_COLUMN,
    )

BASE_DIR = Path(__file__).resolve().parent.parent


def compute_permutation_importance(
    model: keras.Model,
    pipeline,
    X_raw: pd.DataFrame,
    y_true: np.ndarray,
    baseline_auc: float | None = None,
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Measures permutation feature influence by shuffling each raw feature
    across n_repeats, passing through the full preprocessing pipeline,
    and calculating the mean decrease in ROC-AUC.

    Returns:
        pd.DataFrame: Features sorted by descending mean importance.
    """
    rng = np.random.RandomState(seed)
    features = list(X_raw.columns)

    if baseline_auc is None:
        X_proc_base = pipeline.transform(X_raw)
        y_prob_base = model.predict(X_proc_base, verbose=0).ravel()
        baseline_auc = float(roc_auc_score(y_true, y_prob_base))

    importance_records = {feat: [] for feat in features}

    for feat in features:
        for _ in range(n_repeats):
            X_shuffled = X_raw.copy()
            X_shuffled[feat] = rng.permutation(X_shuffled[feat].values)

            X_proc_shuffled = pipeline.transform(X_shuffled)
            y_prob_shuffled = model.predict(X_proc_shuffled, verbose=0).ravel()
            auc_shuffled = float(roc_auc_score(y_true, y_prob_shuffled))

            score_drop = max(0.0, baseline_auc - auc_shuffled)
            importance_records[feat].append(score_drop)

    df_imp = pd.DataFrame(
        {
            "Feature": features,
            "MeanImportance": [np.mean(importance_records[f]) for f in features],
            "StdImportance": [np.std(importance_records[f]) for f in features],
        }
    ).sort_values(by="MeanImportance", ascending=True)

    return df_imp


def plot_feature_importance(
    df_importance: pd.DataFrame,
    df_raw: pd.DataFrame,
    save_path: Path,
) -> None:
    """
    Generates a 2-panel explainability figure:
    - Panel 1: Permutation feature influence ranking with standard error bars.
    - Panel 2: Pearson correlation of each clinical attribute with the target.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    features = df_importance["Feature"].tolist()
    correlations = df_raw[features].apply(lambda c: c.corr(df_raw[TARGET_COLUMN]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Feature Influence (Drop in ROC-AUC)
    y_pos = np.arange(len(df_importance))
    ax1.barh(
        y_pos,
        df_importance["MeanImportance"],
        xerr=df_importance["StdImportance"],
        color="#2563eb",
        alpha=0.85,
        capsize=4,
        edgecolor="#1e3a8a",
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_importance["Feature"], fontsize=11, fontweight="bold")
    ax1.set_xlabel("Mean Decrease in ROC-AUC (Feature Influence)", fontsize=11)
    ax1.set_title("ANN Feature Influence on Model Prediction (Test Set)", fontsize=13, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)

    for i, (m_val, s_val) in enumerate(zip(df_importance["MeanImportance"], df_importance["StdImportance"])):
        ax1.text(m_val + s_val + 0.002, i, f"{m_val:.4f}", va="center", fontsize=9, color="#1e3a8a")

    # Panel 2: Pearson Correlation with Target
    bar_colors = ["#dc2626" if val > 0 else "#2563eb" for val in correlations]
    ax2.barh(y_pos, correlations.values, color=bar_colors, alpha=0.85, edgecolor="#333333")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_importance["Feature"], fontsize=11, fontweight="bold")
    ax2.axvline(0, color="#666666", linestyle="--", linewidth=1)
    ax2.set_xlabel("Pearson Correlation with Target (0 = No Disease, 1 = Disease)", fontsize=11)
    ax2.set_title("Clinical Feature Correlation with Target", fontsize=13, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    plt.suptitle("Model Explainability: Feature Influence & Risk Marker Association", fontsize=15, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.01,
        "Note: Feature influence reflects how strongly the model relies upon an input variable for prediction.\nStatistical association does NOT imply medical or biological causation.",
        ha="center",
        fontsize=10,
        style="italic",
        color="#4b5563",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Explainability] Saved feature influence visualization to: {save_path}")


def run_explainability():
    """CLI runner to compute and save permutation feature influence."""
    model_path = BASE_DIR / "models" / "heart_disease_ann.keras"
    pipeline_path = BASE_DIR / "models" / "preprocessing_pipeline.pkl"
    results_dir = BASE_DIR / "results"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}. Train the model first.")
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline file not found at: {pipeline_path}. Run preprocessing first.")

    df_raw = load_raw_data()
    X_raw = df_raw[ALL_FEATURE_COLUMNS].copy()
    y_raw = df_raw[TARGET_COLUMN].copy().values

    # Test split matching evaluation
    _, X_test_raw, _, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )

    model = keras.models.load_model(model_path)
    pipeline = joblib.load(pipeline_path)

    print("Computing permutation feature influence on held-out test data...")
    df_imp = compute_permutation_importance(model, pipeline, X_test_raw, y_test)
    plot_feature_importance(df_imp, df_raw, results_dir / "feature_analysis.png")
    print("Explainability analysis completed successfully.")
    return df_imp


if __name__ == "__main__":
    run_explainability()
