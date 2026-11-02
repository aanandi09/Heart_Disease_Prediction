"""
Preprocessing Module for Heart Disease Prediction using ANN
===========================================================
Handles data loading, feature definition, column transformations,
and train/validation/test splitting with stratification.

Supports:
1. Leakage-Free Mode (deduplicate=True): Evaluates on unique patient records (N=302)
   to ensure zero sample contamination between train, validation, and test partitions.
2. Standard Benchmark Mode (deduplicate=False): Standard 1025-row Kaggle format for
   comparative baseline reproducibility.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Feature categorizations based on heart.csv structure
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
TARGET_COLUMN = "target"


def get_default_dataset_path() -> Path:
    """Returns absolute path to dataset/heart.csv."""
    return BASE_DIR / "dataset" / "heart.csv"


def get_default_pipeline_path() -> Path:
    """Returns absolute path to models/preprocessing_pipeline.pkl."""
    return BASE_DIR / "models" / "preprocessing_pipeline.pkl"


def load_raw_data(
    csv_path: Path | str | None = None,
    deduplicate: bool = False,
) -> pd.DataFrame:
    """
    Loads raw heart disease dataset from CSV file.
    Does NOT modify or replace the underlying file on disk.
    
    Args:
        csv_path: Path to dataset CSV file.
        deduplicate: If True, drops duplicate patient records in-memory (N=302).
    """
    if csv_path is None:
        csv_path = get_default_dataset_path()
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    if deduplicate:
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a Scikit-Learn ColumnTransformer pipeline.
    - Scales numerical features with StandardScaler.
    - One-hot encodes multi-class categorical features.
    - Passes binary features (0/1) through without distortion.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "bin",
                "passthrough",
                BINARY_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_leakage_free_splits(
    data_path: Path | str | None = None,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
    save_pipeline: bool = True,
    pipeline_path: Path | str | None = None,
):
    """
    Creates a rigorous 3-way split (Train / Validation / Test) on DEDUPLICATED data (N=302).
    Zero patient overlap exists across Train, Validation, and Test partitions.
    
    - Preprocessor is fitted STRICTLY on X_train (No Data Leakage).
    - Validation set is reserved for training callbacks & threshold tuning.
    - Test set is reserved exclusively for final unbiased reporting.
    """
    df = load_raw_data(data_path, deduplicate=True)

    required_cols = ALL_FEATURE_COLUMNS + [TARGET_COLUMN]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset.")

    X = df[ALL_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy().values

    # 1. Split off test set (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 2. Split train into train and validation (val_size of remaining)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    # 3. Fit ColumnTransformer ONLY on X_train
    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # 4. Save fitted pipeline
    if save_pipeline:
        if pipeline_path is None:
            pipeline_path = get_default_pipeline_path()
        pipeline_path = Path(pipeline_path)
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, pipeline_path)
        print(f"[Preprocessing] Saved leakage-free preprocessor to: {pipeline_path}")

    return X_train_proc, X_val_proc, X_test_proc, y_train, y_val, y_test, preprocessor


def preprocess_and_split(
    data_path: Path | str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    deduplicate: bool = True,
    save_pipeline: bool = True,
    pipeline_path: Path | str | None = None,
):
    """
    Standard stratified train/test split.
    Defaults to deduplicate=True for scientifically valid zero-leakage evaluation.
    """
    df = load_raw_data(data_path, deduplicate=deduplicate)

    required_cols = ALL_FEATURE_COLUMNS + [TARGET_COLUMN]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset.")

    X = df[ALL_FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy().values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    if save_pipeline:
        if pipeline_path is None:
            pipeline_path = get_default_pipeline_path()
        pipeline_path = Path(pipeline_path)
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, pipeline_path)
        print(f"[Preprocessing] Saved fitted preprocessor to: {pipeline_path}")

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor


if __name__ == "__main__":
    print("=" * 70)
    print("Verifying Leakage-Free Data Preprocessing Pipeline")
    print("=" * 70)
    X_tr, X_va, X_te, y_tr, y_va, y_te, prep = get_leakage_free_splits()
    print(f"X_train shape : {X_tr.shape} (Positives: {np.sum(y_tr)}, Negatives: {len(y_tr)-np.sum(y_tr)})")
    print(f"X_val shape   : {X_va.shape} (Positives: {np.sum(y_va)}, Negatives: {len(y_va)-np.sum(y_va)})")
    print(f"X_test shape  : {X_te.shape} (Positives: {np.sum(y_te)}, Negatives: {len(y_te)-np.sum(y_te)})")
    print("Zero train-test leakage guaranteed across unique patient cohorts.")
