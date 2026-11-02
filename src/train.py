"""
ANN Model Training & Comparative Benchmarking Module
=====================================================
Orchestrates training of Baseline and Improved Artificial Neural Networks (ANN),
along with benchmark machine learning baselines:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Baseline ANN
5. Improved ANN (Final Chosen Model)

Evaluation Methodology:
- Strict Zero-Leakage Split: Deduplicated patient records (N=302) partitioned into
  Train (N=204), Validation (N=37), and Held-Out Test (N=61).
- Preprocessing pipeline fitted STRICTLY on X_train.
- Validation partition reserved for early stopping and hyperparameter monitoring.
- Held-out test set evaluated strictly once at the end.
- Standard 1,025-sample benchmark also recorded for literature comparison.

Saves:
- models/heart_disease_ann.keras
- models/preprocessing_pipeline.pkl
- results/model_comparison.csv
- results/model_comparison_standard.csv
- results/model_comparison.png
- results/training_history.png
- results/ann_training_history.png
"""

import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, callbacks

try:
    from src.preprocess import (
        get_leakage_free_splits,
        preprocess_and_split,
        build_preprocessor,
        load_raw_data,
        ALL_FEATURE_COLUMNS,
        TARGET_COLUMN,
    )
except ModuleNotFoundError:
    from preprocess import (
        get_leakage_free_splits,
        preprocess_and_split,
        build_preprocessor,
        load_raw_data,
        ALL_FEATURE_COLUMNS,
        TARGET_COLUMN,
    )

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent


def set_seed(seed: int = 42):
    """Sets random seeds for strict computational reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_baseline_ann(input_dim: int) -> keras.Model:
    """Constructs a simple baseline 2-layer MLP without regularization."""
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,), name="input_layer"),
            layers.Dense(32, activation="relu", name="dense_1"),
            layers.Dense(16, activation="relu", name="dense_2"),
            layers.Dense(1, activation="sigmoid", name="output_layer"),
        ],
        name="Baseline_ANN",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_improved_ann(input_dim: int) -> keras.Model:
    """
    Constructs an optimized, regularized deep MLP:
    - Input dimension: 27 features
    - Dense 48 (ReLU + L2 weight decay 0.001)
    - Batch Normalization
    - Dropout (0.25)
    - Dense 24 (ReLU + L2 weight decay 0.001)
    - Batch Normalization
    - Dropout (0.20)
    - Dense 12 (ReLU)
    - Dropout (0.10)
    - Sigmoid output for calibrated risk probability estimation
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,), name="input_layer"),
            layers.Dense(
                48,
                activation="relu",
                kernel_regularizer=regularizers.l2(0.001),
                name="dense_1",
            ),
            layers.BatchNormalization(name="batch_norm_1"),
            layers.Dropout(0.25, name="dropout_1"),
            layers.Dense(
                24,
                activation="relu",
                kernel_regularizer=regularizers.l2(0.001),
                name="dense_2",
            ),
            layers.BatchNormalization(name="batch_norm_2"),
            layers.Dropout(0.20, name="dropout_2"),
            layers.Dense(12, activation="relu", name="dense_3"),
            layers.Dropout(0.10, name="dropout_3"),
            layers.Dense(1, activation="sigmoid", name="output_layer"),
        ],
        name="Improved_Heart_Disease_ANN",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_model_predictions(
    name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict:
    """Calculates standardized binary classification metrics."""
    return {
        "Model": name,
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1-Score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def plot_comparison_chart(df_comp: pd.DataFrame, save_path: Path):
    """Generates a high-resolution bar plot comparing all model metrics."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    df_melted = df_comp.melt(
        id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score"
    )

    plt.figure(figsize=(13, 6))
    palette = sns.color_palette("Set2", len(metrics))
    ax = sns.barplot(
        data=df_melted, x="Model", y="Score", hue="Metric", palette=palette
    )

    plt.title(
        "Benchmark Model Comparison on Leakage-Free Held-Out Test Set (N=61)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.ylabel("Performance Score (0.0 - 1.0)", fontsize=11)
    plt.xlabel("Evaluated Machine Learning Models", fontsize=11)
    plt.ylim(0.60, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True, fontsize=10)

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=7.5,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Training] Saved model comparison chart to: {save_path}")


def plot_training_history(history, save_path: Path, second_path: Path | None = None):
    """Visualizes training & validation loss and accuracy progression curves."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(
        epochs,
        history.history["loss"],
        label="Training Loss",
        color="#1e3a8a",
        linewidth=2,
    )
    axes[0].plot(
        epochs,
        history.history["val_loss"],
        label="Validation Loss",
        color="#0d9488",
        linewidth=2,
        linestyle="--",
    )
    axes[0].set_title("ANN Model Loss vs Epochs", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epochs", fontsize=11)
    axes[0].set_ylabel("Binary Crossentropy Loss", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy curve
    axes[1].plot(
        epochs,
        history.history["accuracy"],
        label="Training Accuracy",
        color="#16a34a",
        linewidth=2,
    )
    axes[1].plot(
        epochs,
        history.history["val_accuracy"],
        label="Validation Accuracy",
        color="#ea580c",
        linewidth=2,
        linestyle="--",
    )
    axes[1].set_title("ANN Model Accuracy vs Epochs", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epochs", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    if second_path:
        second_path = Path(second_path)
        plt.savefig(second_path, dpi=300)
    plt.close()
    print(f"[Training] Saved training history plot to: {save_path}")


def train_and_compare_all():
    """
    Executes the end-to-end training and comparative benchmarking pipeline:
    1. Zero-leakage preprocessing on unique patients (N=302).
    2. Training classical baseline models (Logistic Regression, Decision Tree, Random Forest).
    3. Training Baseline ANN vs Improved Regularized ANN.
    4. Evaluating all models on the held-out test partition (N=61).
    5. Saving comparison table, plots, and optimal model weights.
    6. Generating secondary 1,025-sample benchmark for literature comparison.
    """
    set_seed(42)
    results_dir = BASE_DIR / "results"
    models_dir = BASE_DIR / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 2: Leakage-Free Data Preprocessing & Model Benchmarking")
    print("=" * 70)

    # 1. Rigorous 3-way split on unique patients (N=302)
    pipeline_path = models_dir / "preprocessing_pipeline.pkl"
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        preprocessor,
    ) = get_leakage_free_splits(
        test_size=0.20,
        val_size=0.15,
        random_state=42,
        save_pipeline=True,
        pipeline_path=pipeline_path,
    )

    input_dim = X_train.shape[1]
    print(f"Preprocessed Feature Matrix Dimension: {input_dim} columns")
    print(f"Training partition   : {X_train.shape[0]} samples (Positives: {np.sum(y_train)})")
    print(f"Validation partition : {X_val.shape[0]} samples (Positives: {np.sum(y_val)})")
    print(f"Held-Out Test cohort : {X_test.shape[0]} samples (Positives: {np.sum(y_test)})")

    comparison_results = []

    # 1. Logistic Regression
    print("\n--> Training 1/5: Logistic Regression Baseline...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    lr_res = evaluate_model_predictions("Logistic Regression", y_test, lr_pred, lr_prob)
    comparison_results.append(lr_res)

    # 2. Decision Tree
    print("--> Training 2/5: Decision Tree Classifier...")
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_prob = dt.predict_proba(X_test)[:, 1]
    dt_res = evaluate_model_predictions("Decision Tree", y_test, dt_pred, dt_prob)
    comparison_results.append(dt_res)

    # 3. Random Forest
    print("--> Training 3/5: Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_res = evaluate_model_predictions("Random Forest", y_test, rf_pred, rf_prob)
    comparison_results.append(rf_res)

    # 4. Baseline ANN
    print("--> Training 4/5: Baseline Artificial Neural Network (MLP)...")
    baseline_ann = build_baseline_ann(input_dim)
    baseline_ann.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        verbose=0,
    )
    b_prob = baseline_ann.predict(X_test, verbose=0).ravel()
    b_pred = (b_prob >= 0.5).astype(int)
    b_res = evaluate_model_predictions("Baseline ANN", y_test, b_pred, b_prob)
    comparison_results.append(b_res)

    # 5. Improved Regularized ANN (Selected Final Model)
    print("--> Training 5/5: Improved Regularized ANN (Batch Norm + Dropout + L2)...")
    improved_ann = build_improved_ann(input_dim)
    ann_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history = improved_ann.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=120,
        batch_size=16,
        callbacks=ann_callbacks,
        verbose=1,
    )
    imp_prob = improved_ann.predict(X_test, verbose=0).ravel()
    imp_pred = (imp_prob >= 0.5).astype(int)
    imp_res = evaluate_model_predictions(
        "Improved ANN (Selected)", y_test, imp_pred, imp_prob
    )
    comparison_results.append(imp_res)

    # Save Leakage-Free Comparison DataFrame
    df_comparison = pd.DataFrame(comparison_results)
    csv_path = results_dir / "model_comparison.csv"
    df_comparison.to_csv(csv_path, index=False)
    print(f"\n[Training] Saved leakage-free model comparison table to: {csv_path}")

    print("\n" + "=" * 70)
    print("LEAKAGE-FREE TEST SET BENCHMARK RESULTS (Unique Patients N=61)")
    print("=" * 70)
    print(df_comparison.to_string(index=False))

    # Save Comparison Chart
    plot_comparison_chart(df_comparison, results_dir / "model_comparison.png")

    # Save Training History Plots
    plot_training_history(
        history,
        results_dir / "training_history.png",
        results_dir / "ann_training_history.png",
    )

    # Save Final Improved ANN Model
    model_save_path = models_dir / "heart_disease_ann.keras"
    improved_ann.save(model_save_path)
    print(f"[Training] Saved final ANN model to: {model_save_path}")

    # -------------------------------------------------------------------------
    # Secondary Benchmark: Standard 1,025-Row Split for Literature Context
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECONDARY BENCHMARK: Full 1,025-Row Dataset (Kaggle Baseline Context)")
    print("=" * 70)
    df_full = load_raw_data(deduplicate=False)
    X_f = df_full[ALL_FEATURE_COLUMNS].copy()
    y_f = df_full[TARGET_COLUMN].copy().values
    X_f_tr, X_f_te, y_f_tr, y_f_te = train_test_split(
        X_f, y_f, test_size=0.20, random_state=42, stratify=y_f
    )
    prep_f = build_preprocessor()
    X_f_tr_proc = prep_f.fit_transform(X_f_tr)
    X_f_te_proc = prep_f.transform(X_f_te)

    std_results = []
    for m_name, m_model in [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
    ]:
        m_model.fit(X_f_tr_proc, y_f_tr)
        m_p = m_model.predict(X_f_te_proc)
        m_prob = m_model.predict_proba(X_f_te_proc)[:, 1]
        std_results.append(evaluate_model_predictions(m_name, y_f_te, m_p, m_prob))

    # Full ANN
    ann_std = build_improved_ann(input_dim)
    ann_std.fit(
        X_f_tr_proc,
        y_f_tr,
        validation_split=0.15,
        epochs=60,
        batch_size=32,
        verbose=0,
    )
    p_std = ann_std.predict(X_f_te_proc, verbose=0).ravel()
    std_results.append(
        evaluate_model_predictions(
            "Improved ANN (Selected)", y_f_te, (p_std >= 0.5).astype(int), p_std
        )
    )

    df_std = pd.DataFrame(std_results)
    std_csv_path = results_dir / "model_comparison_standard.csv"
    df_std.to_csv(std_csv_path, index=False)
    print(f"[Training] Saved standard 1,025 benchmark comparison to: {std_csv_path}")

    return df_comparison, improved_ann


if __name__ == "__main__":
    train_and_compare_all()
