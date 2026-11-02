"""
Streamlit Web Application: Heart Disease Prediction using ANN
==============================================================
An academic, research-oriented, and presentation-ready B.Tech AI/ML dashboard.

Main Sections:
1. Clinical Prediction (13 clinical inputs, predicted probability, threshold, session history, downloadable report)
2. Model Performance (Test metrics, confusion matrix, ROC curve)
3. Model Comparison (Benchmark comparisons against classical ML models)
4. Explainable AI (Permutation feature influence on model prediction)
5. Dataset Insights (Exploratory data analysis from dataset/heart.csv)
6. Methodology & Training (Architecture, loss curves, early stopping, and ethical disclaimers)
"""

from datetime import datetime
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow import keras

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction | ANN System",
    page_icon="Cardiology",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Relative path definitions from project root
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "heart.csv"
MODEL_PATH = BASE_DIR / "models" / "heart_disease_ann.keras"
PIPELINE_PATH = BASE_DIR / "models" / "preprocessing_pipeline.pkl"
RESULTS_DIR = BASE_DIR / "results"
METRICS_JSON_PATH = RESULTS_DIR / "final_metrics.json"
COMPARISON_CSV_PATH = RESULTS_DIR / "model_comparison.csv"


# -----------------------------------------------------------------------------
# Cached Artifact Loaders with Safe Fallbacks
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ml_artifacts():
    """Loads and caches the trained Keras ANN model and Scikit-Learn preprocessor."""
    if not MODEL_PATH.exists():
        return None, None, f"Model file not found at: {MODEL_PATH}"
    if not PIPELINE_PATH.exists():
        return None, None, f"Pipeline file not found at: {PIPELINE_PATH}"

    try:
        model = keras.models.load_model(MODEL_PATH)
        pipeline = joblib.load(PIPELINE_PATH)
        return model, pipeline, None
    except Exception as e:
        return None, None, f"Failed to load ML artifacts: {e}"


@st.cache_data
def load_dataset():
    """Loads and caches the raw dataset."""
    if not DATASET_PATH.exists():
        return None, f"Dataset not found at: {DATASET_PATH}"
    try:
        df = pd.read_csv(DATASET_PATH)
        return df, None
    except Exception as e:
        return None, f"Error reading dataset: {e}"


@st.cache_data
def load_benchmark_metrics():
    """Loads test metrics and model comparison records."""
    metrics = {}
    if METRICS_JSON_PATH.exists():
        try:
            with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}

    comp_df = pd.DataFrame()
    if COMPARISON_CSV_PATH.exists():
        try:
            comp_df = pd.read_csv(COMPARISON_CSV_PATH)
        except Exception:
            comp_df = pd.DataFrame()

    return metrics, comp_df


# -----------------------------------------------------------------------------
# Clean Academic Medical-Tech CSS
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .header-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.15rem;
    }
    .header-sub {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.1rem;
    }
    .disclaimer-card {
        background-color: #fefce8;
        border-left: 4px solid #ca8a04;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 0.92rem;
        color: #854d0e;
        margin-bottom: 1.4rem;
        line-height: 1.5;
    }
    .overview-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 1.5rem;
    }
    .card-high {
        background-color: #fef2f2;
        border: 1.5px solid #f87171;
        border-radius: 8px;
        padding: 18px;
        margin-top: 15px;
    }
    .card-low {
        background-color: #f0fdf4;
        border: 1.5px solid #4ade80;
        border-radius: 8px;
        padding: 18px;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
def main():
    # Session state initialization for prediction history
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    # 1. Header
    st.markdown('<div class="header-title">Heart Disease Prediction using Artificial Neural Network</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="header-sub">An Artificial Neural Network based binary classification system for cardiovascular risk-marker prediction.</div>',
        unsafe_allow_html=True,
    )

    # Educational Disclaimer
    st.markdown(
        """
        <div class="disclaimer-card">
            <strong>Educational & Research Disclaimer:</strong><br>
            This software is an educational machine learning project and is <strong>not</strong> a certified medical diagnostic system.
            Predictions reflect statistical risk associations learned from the dataset and must not be used as a substitute for professional
            clinical judgment, healthcare diagnosis, or medical treatment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Model Overview Card
    st.markdown(
        """
        <div class="overview-card">
            <table style="width:100%; font-size:0.95rem; border-collapse:collapse;">
                <tr>
                    <td style="width:25%;"><strong>Model Architecture:</strong> Multi-Layer Perceptron (ANN)</td>
                    <td style="width:25%;"><strong>Prediction Task:</strong> Binary Classification</td>
                    <td style="width:25%;"><strong>Input Features:</strong> 13 Clinical Attributes</td>
                    <td style="width:25%;"><strong>Dataset Source:</strong> Heart Disease Dataset (1,025 Records)</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load Artifacts
    model, pipeline, err_art = load_ml_artifacts()
    df, err_df = load_dataset()
    metrics, comp_df = load_benchmark_metrics()

    if err_art:
        st.error(err_art)
        st.info("Run `python src/train.py` from the project root to generate the required model artifacts.")
        return

    # Navigation Tabs
    tab_pred, tab_perf, tab_comp, tab_xai, tab_eda, tab_method = st.tabs(
        [
            "Clinical Prediction",
            "Model Performance",
            "Model Comparison",
            "Explainable AI",
            "Dataset Insights",
            "Project Methodology",
        ]
    )

    # =========================================================================
    # TAB 1: CLINICAL PREDICTION
    # =========================================================================
    with tab_pred:
        st.subheader("Patient Clinical Input")
        st.write("Enter the patient's 13 physiological parameters across the four categories below:")

        with st.form("prediction_input_form"):
            c1, c2, c3, c4 = st.columns(4)

            # Section 1: Demographic Information
            with c1:
                st.markdown("#### 1. Demographics")
                age = st.number_input(
                    "Age (years)",
                    min_value=20,
                    max_value=100,
                    value=55,
                    step=1,
                    help="Patient age at examination (20-100)",
                )
                sex_choice = st.selectbox(
                    "Biological Sex",
                    options=["Male", "Female"],
                    index=0,
                    help="Biological sex of the patient",
                )
                sex = 1 if sex_choice == "Male" else 0

            # Section 2: Cardiovascular Measurements
            with c2:
                st.markdown("#### 2. Vitals")
                trestbps = st.number_input(
                    "Resting Blood Pressure (mm Hg)",
                    min_value=80,
                    max_value=220,
                    value=130,
                    step=1,
                    help="Resting blood pressure on hospital admission (mm Hg)",
                )
                chol = st.number_input(
                    "Serum Cholesterol (mg/dl)",
                    min_value=100,
                    max_value=600,
                    value=240,
                    step=1,
                    help="Serum cholesterol level (mg/dl)",
                )
                thalach = st.number_input(
                    "Maximum Heart Rate (bpm)",
                    min_value=60,
                    max_value=220,
                    value=150,
                    step=1,
                    help="Peak heart rate achieved during exercise stress testing",
                )

            # Section 3: Clinical Indicators
            with c3:
                st.markdown("#### 3. Clinical Indicators")
                cp_map = {
                    "Typical Angina (0)": 0,
                    "Atypical Angina (1)": 1,
                    "Non-anginal Pain (2)": 2,
                    "Asymptomatic (3)": 3,
                }
                cp_selected = st.selectbox(
                    "Chest Pain Type (cp)",
                    options=list(cp_map.keys()),
                    index=0,
                    help="Chest pain category (0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic)",
                )
                cp = cp_map[cp_selected]

                fbs_choice = st.selectbox(
                    "Fasting Blood Sugar > 120 mg/dl (fbs)",
                    options=["No (<= 120 mg/dl)", "Yes (> 120 mg/dl)"],
                    index=0,
                    help="Fasting blood sugar above 120 mg/dl",
                )
                fbs = 1 if "Yes" in fbs_choice else 0

                exang_choice = st.selectbox(
                    "Exercise-Induced Angina (exang)",
                    options=["No (0)", "Yes (1)"],
                    index=0,
                    help="Whether exercise induces angina chest discomfort",
                )
                exang = 1 if "Yes" in exang_choice else 0

            # Section 4: ECG / Diagnostic Features
            with c4:
                st.markdown("#### 4. ECG & Diagnostics")
                restecg_map = {
                    "Normal (0)": 0,
                    "ST-T Wave Abnormality (1)": 1,
                    "Left Ventricular Hypertrophy (2)": 2,
                }
                restecg_selected = st.selectbox(
                    "Resting ECG (restecg)",
                    options=list(restecg_map.keys()),
                    index=0,
                    help="Resting electrocardiogram results",
                )
                restecg = restecg_map[restecg_selected]

                oldpeak = st.number_input(
                    "ST Depression (oldpeak)",
                    min_value=0.0,
                    max_value=6.5,
                    value=1.0,
                    step=0.1,
                    format="%.1f",
                    help="ST depression induced by exercise relative to rest (in mm)",
                )

                slope_map = {
                    "Upsloping (0)": 0,
                    "Flat (1)": 1,
                    "Downsloping (2)": 2,
                }
                slope_selected = st.selectbox(
                    "Peak Exercise ST Slope (slope)",
                    options=list(slope_map.keys()),
                    index=1,
                    help="Slope of peak exercise ST segment",
                )
                slope = slope_map[slope_selected]

                ca = st.selectbox(
                    "Number of Major Vessels (ca)",
                    options=[0, 1, 2, 3, 4],
                    index=0,
                    help="Number of major vessels (0-4) colored by fluoroscopy",
                )

                thal_map = {
                    "Unknown / Null (0)": 0,
                    "Normal Blood Flow (1)": 1,
                    "Fixed Defect (2)": 2,
                    "Reversible Defect (3)": 3,
                }
                thal_selected = st.selectbox(
                    "Thalassemia Category (thal)",
                    options=list(thal_map.keys()),
                    index=2,
                    help="Thalassemia cardiac blood flow disorder status",
                )
                thal = thal_map[thal_selected]

            submit_prediction = st.form_submit_button("Generate Prediction", use_container_width=True)

        # Handle Prediction Inference
        if submit_prediction:
            input_record = {
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal,
            }
            input_df = pd.DataFrame([input_record])

            try:
                # Apply saved preprocessing pipeline
                X_transformed = pipeline.transform(input_df)

                # Generate probability via ANN
                pred_prob = float(model.predict(X_transformed, verbose=0)[0][0])
                pred_class = int(pred_prob >= 0.5)

                st.divider()
                st.subheader("Prediction Result")

                # Metrics Row
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                with m_c1:
                    st.metric("Predicted Class", f"Class {pred_class}")
                with m_c2:
                    st.metric("Predicted Probability", f"{pred_prob * 100:.2f}%")
                with m_c3:
                    st.metric("Classification Threshold", "50.00%")
                with m_c4:
                    st.metric("Statistical Risk Category", "Elevated Risk" if pred_class == 1 else "Lower Risk")

                # Probability Scale
                st.write("**Model Probability Scale:**")
                st.progress(pred_prob)

                # Result Explanation Card
                if pred_class == 1:
                    st.markdown(
                        f"""
                        <div class="card-high">
                            <h4 style="color:#b91c1c; margin:0 0 8px 0;">Prediction: Higher Predicted Likelihood of Heart Disease (Class 1)</h4>
                            <p style="color:#7f1d1d; margin:0; line-height:1.5;">
                                The model estimates a <strong>{pred_prob * 100:.2f}%</strong> predicted probability for the positive class
                                based on the supplied input features.<br>
                                <em>This indicates that the patient's physiological markers align with patterns associated with cardiovascular risk in the training cohort.</em>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="card-low">
                            <h4 style="color:#15803d; margin:0 0 8px 0;">Prediction: Lower Predicted Likelihood of Heart Disease (Class 0)</h4>
                            <p style="color:#14532d; margin:0; line-height:1.5;">
                                The model estimates a <strong>{(1.0 - pred_prob) * 100:.2f}%</strong> predicted probability for the negative class
                                based on the supplied input features.<br>
                                <em>This indicates that the patient's physiological markers align with low-risk patterns in the training cohort.</em>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Record into session prediction history
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                history_entry = {
                    "Timestamp": now_str,
                    "Age": age,
                    "Sex": "Male" if sex == 1 else "Female",
                    "Chest Pain": cp_selected.split(" ")[0],
                    "Resting BP": trestbps,
                    "Cholesterol": chol,
                    "Max HR": thalach,
                    "Predicted Class": pred_class,
                    "Predicted Probability": f"{pred_prob * 100:.2f}%",
                }
                st.session_state.prediction_history.insert(0, history_entry)

                # Downloadable Summary
                report_content = f"""HEART DISEASE PREDICTION REPORT (ACADEMIC ML SYSTEM)
==================================================
Timestamp: {now_str}
Model Architecture: Artificial Neural Network (Multi-Layer Perceptron)
Classification Task: Binary Heart Disease Risk Prediction

PREDICTION OUTPUT:
------------------
Predicted Class: Class {pred_class} ({"Higher Likelihood" if pred_class == 1 else "Lower Likelihood"})
Predicted Probability: {pred_prob * 100:.2f}%
Classification Threshold: 50.00%

INPUT CLINICAL FEATURES:
------------------------
1. Age: {age} years
2. Sex: {"Male" if sex == 1 else "Female"}
3. Chest Pain Type: {cp_selected}
4. Resting Blood Pressure: {trestbps} mm Hg
5. Serum Cholesterol: {chol} mg/dl
6. Fasting Blood Sugar > 120 mg/dl: {fbs_choice}
7. Resting ECG: {restecg_selected}
8. Maximum Heart Rate: {thalach} bpm
9. Exercise-Induced Angina: {exang_choice}
10. ST Depression (oldpeak): {oldpeak} mm
11. Peak ST Slope: {slope_selected}
12. Number of Major Vessels: {ca}
13. Thalassemia Category: {thal_selected}

DISCLAIMER:
-----------
This report was generated by an educational machine learning project for research purposes only.
It is NOT a medical diagnosis and must NOT be used as a substitute for professional medical advice.
"""
                st.download_button(
                    label="Download Prediction Summary (TXT)",
                    data=report_content,
                    file_name=f"heart_prediction_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            except Exception as e:
                st.error(f"Prediction execution error: {e}")

        # Prediction History (Session-based)
        st.divider()
        st.subheader("Session Prediction History")
        st.caption("Temporary session storage only. No personal or health data is permanently stored on disk.")

        if st.session_state.prediction_history:
            df_hist = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(df_hist, use_container_width=True)
            if st.button("Clear Prediction History"):
                st.session_state.prediction_history = []
                st.rerun()
        else:
            st.info("No predictions generated yet in this session.")

    # =========================================================================
    # TAB 2: MODEL PERFORMANCE
    # =========================================================================
    with tab_perf:
        st.subheader("Test-Set Performance Metrics")
        st.caption("Evaluated on 205 held-out patient samples with stratified 80/20 train/test splitting.")

        if metrics:
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Test Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f}%")
            k2.metric("Precision", f"{metrics.get('precision', 0) * 100:.2f}%")
            k3.metric("Recall (Sensitivity)", f"{metrics.get('recall', 0) * 100:.2f}%")
            k4.metric("F1 Score", f"{metrics.get('f1_score', 0) * 100:.2f}%")
            k5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")

            cm_dict = metrics.get("confusion_matrix", {})
            st.markdown(
                f"""
                <div class="overview-card" style="margin-top:15px;">
                    <strong>Confusion Matrix Counts:</strong> &nbsp;
                    True Negatives: <strong>{cm_dict.get('true_negatives', 0)}</strong> &nbsp;|&nbsp;
                    False Positives: <strong>{cm_dict.get('false_positives', 0)}</strong> &nbsp;|&nbsp;
                    False Negatives: <strong>{cm_dict.get('false_negatives', 0)}</strong> &nbsp;|&nbsp;
                    True Positives: <strong>{cm_dict.get('true_positives', 0)}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # Confusion Matrix & ROC Curve Side by Side
        st.subheader("Diagnostic Evaluation Curves")
        c1, c2 = st.columns(2)
        with c1:
            cm_img = RESULTS_DIR / "confusion_matrix.png"
            if cm_img.exists():
                st.image(str(cm_img), caption="Confusion Matrix on Unseen Test Records", use_container_width=True)
            else:
                st.warning("Confusion matrix plot not found.")
        with c2:
            roc_img = RESULTS_DIR / "roc_curve.png"
            if roc_img.exists():
                st.image(str(roc_img), caption="ROC Curve (Area Under Curve = 0.9928)", use_container_width=True)
            else:
                st.warning("ROC curve plot not found.")

    # =========================================================================
    # TAB 3: MODEL COMPARISON
    # =========================================================================
    with tab_comp:
        st.subheader("Baseline Model Benchmarks")
        st.write("Comparison of conventional machine learning models alongside the Artificial Neural Network on the identical test split:")

        if not comp_df.empty:
            st.dataframe(
                comp_df.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color="#dbeafe"),
                use_container_width=True,
            )
        else:
            st.warning("Model comparison records not found.")

        comp_chart = RESULTS_DIR / "model_comparison.png"
        if comp_chart.exists():
            st.image(str(comp_chart), caption="Comparative Performance Across Evaluated Models", use_container_width=True)

        st.info(
            """
            **Benchmarking Findings:**
            - **Logistic Regression**: Linear baseline providing 87.32% accuracy and 0.9435 ROC-AUC.
            - **Decision Tree**: Non-linear tree baseline yielding 88.78% accuracy.
            - **Random Forest**: Strong ensemble baseline achieving 96.10% accuracy.
            - **Artificial Neural Network (Selected)**: Highest performance with 98.05% accuracy and 0.9928 ROC-AUC.
            """
        )

    # =========================================================================
    # TAB 4: EXPLAINABLE AI
    # =========================================================================
    with tab_xai:
        st.subheader("Explainable AI: Feature Influence on Model Prediction")
        st.write(
            """
            Artificial Neural Networks model complex non-linear feature interactions.
            To understand which clinical markers most heavily influence predictions,
            we compute **Permutation Feature Importance** on the held-out test set.
            """
        )

        feat_chart = RESULTS_DIR / "feature_analysis.png"
        if feat_chart.exists():
            st.image(str(feat_chart), caption="Permutation Feature Influence and Risk Marker Correlation", use_container_width=True)
        else:
            st.warning("Feature analysis visualization not found.")

        st.markdown(
            """
            <div class="overview-card">
                <h4 style="color:#1e3a8a; margin:0 0 8px 0;">Methodology & Interpretation</h4>
                <p style="margin:0 0 8px 0; font-size:0.95rem; line-height:1.5;">
                    <strong>How Permutation Feature Influence is Calculated:</strong><br>
                    Each clinical feature is independently shuffled across 10 repeats on the test set, breaking its relationship with the target.
                    The resulting drop in ROC-AUC quantifies how strongly the model's decision boundary depends upon that attribute.
                </p>
                <p style="margin:0; font-size:0.95rem; line-height:1.5;">
                    <strong>Critical Note:</strong><br>
                    Feature influence quantifies statistical reliance in the model.
                    <strong>Feature influence does NOT imply medical or biological causation.</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 5: DATASET INSIGHTS
    # =========================================================================
    with tab_eda:
        st.subheader("Dataset Analytics & Exploratory Data Analysis")
        st.caption("Data source: `dataset/heart.csv` (1,025 observations across 13 clinical variables).")

        if df is not None:
            ed1, ed2, ed3, ed4 = st.columns(4)
            ed1.metric("Total Observations", f"{len(df):,}")
            ed2.metric("Input Features", "13")
            pos_count = int(df["target"].sum())
            ed3.metric("Disease Class 1", f"{pos_count} ({pos_count / len(df) * 100:.1f}%)")
            neg_count = len(df) - pos_count
            ed4.metric("Non-Disease Class 0", f"{neg_count} ({neg_count / len(df) * 100:.1f}%)")

            st.divider()

            e_c1, e_c2 = st.columns(2)
            with e_c1:
                st.markdown("##### Age Distribution by Disease Status")
                fig_age, ax_age = plt.subplots(figsize=(7, 4))
                sns.histplot(data=df, x="age", hue="target", kde=True, palette=["#3b82f6", "#ef4444"], bins=20, ax=ax_age)
                ax_age.set_title("Patient Age Distribution", fontsize=11, fontweight="bold")
                ax_age.set_xlabel("Age (years)")
                st.pyplot(fig_age)

            with e_c2:
                st.markdown("##### Maximum Heart Rate (thalach) vs Disease")
                fig_hr, ax_hr = plt.subplots(figsize=(7, 4))
                sns.boxplot(data=df, x="target", y="thalach", palette=["#3b82f6", "#ef4444"], ax=ax_hr)
                ax_hr.set_xticklabels(["No Disease (0)", "Disease Present (1)"])
                ax_hr.set_title("Peak Heart Rate vs Disease Status", fontsize=11, fontweight="bold")
                ax_hr.set_ylabel("Max Heart Rate (bpm)")
                st.pyplot(fig_hr)

            st.markdown("##### Raw Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

        elif err_df:
            st.warning(err_df)

    # =========================================================================
    # TAB 6: PROJECT METHODOLOGY & TRAINING ANALYSIS
    # =========================================================================
    with tab_method:
        st.subheader("ANN Training Analysis & Methodology")
        st.caption("Detailed overview of the engineering pipeline, loss curves, and academic design decisions.")

        hist_chart = RESULTS_DIR / "training_history.png"
        if hist_chart.exists():
            st.image(str(hist_chart), caption="Training vs Validation Accuracy & Loss across Epochs", use_container_width=True)

        st.markdown(
            """
            ### Architectural Design Decisions
            1. **Data-Leakage Prevention**:
               Train and test sets are split prior to fitting any transformations (`test_size=0.2, stratify=y`).
               The `ColumnTransformer` is fitted exclusively on training data and serialized to `models/preprocessing_pipeline.pkl`.
            
            2. **Feature Engineering & Encodings**:
               - Continuous features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) are standardized using `StandardScaler`.
               - Nominal features (`cp`, `restecg`, `slope`, `ca`, `thal`) are encoded via `OneHotEncoder(handle_unknown='ignore')`.
               - Binary features (`sex`, `fbs`, `exang`) pass through directly without artificial scaling.
               - Transformed feature vector: **27 features**.

            3. **Neural Network Architecture**:
               - Dense Layer (64 neurons, ReLU activation, L2 weight decay $\\lambda = 0.001$)
               - Batch Normalization Layer (stabilizing internal covariate shift)
               - Dropout Layer (rate = 0.30)
               - Dense Layer (32 neurons, ReLU activation, L2 weight decay $\\lambda = 0.001$)
               - Batch Normalization Layer
               - Dropout Layer (rate = 0.20)
               - Dense Layer (16 neurons, ReLU activation)
               - Dropout Layer (rate = 0.10)
               - Output Layer (1 neuron, Sigmoid activation)

            4. **Regularization & Stopping Criteria**:
               - Monitored `val_loss` using `EarlyStopping(patience=20, restore_best_weights=True)`.
               - Learning rate scheduler `ReduceLROnPlateau(factor=0.5, patience=6)`.
               - Training halted at epoch 58, successfully restoring optimal model weights from epoch 38.

            5. **Limitations & Future Scope**:
               - Dataset derives from standard Cleveland cohort repetitions (1,025 entries). Multi-center validation is required for real-world application.
               - Static cross-sectional features do not capture longitudinal patient history.
            """
        )


if __name__ == "__main__":
    main()
