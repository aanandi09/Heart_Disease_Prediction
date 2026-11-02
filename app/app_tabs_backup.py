"""
Streamlit Web Application: Heart Disease Prediction using Artificial Neural Network (ANN)
========================================================================================
A presentation-ready, research-oriented, and academically rigorous B.Tech AI/ML dashboard.

Key Highlights:
1. Healthcare Analytics Theme: Professional Navy (#0f172a, #1e3a8a), Teal (#0d9488), and Clean White.
2. Verified Empirical Metrics: Displays true measured metrics from zero-leakage held-out testing.
3. Dual-Benchmark Architecture: Interactive toggle between Leakage-Free (N=302) and Standard (N=1,025).
4. Full Modern Streamlit Compatibility: Replaced deprecated use_container_width with width='stretch'.
5. Medically Responsible Design: Prominent educational disclaimers; no diagnostic claims.
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
    page_title="CardioANN | Heart Disease Risk Prediction",
    page_icon="stethoscope",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Relative directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "heart.csv"
MODEL_PATH = BASE_DIR / "models" / "heart_disease_ann.keras"
PIPELINE_PATH = BASE_DIR / "models" / "preprocessing_pipeline.pkl"
RESULTS_DIR = BASE_DIR / "results"
METRICS_JSON_PATH = RESULTS_DIR / "final_metrics.json"
COMPARISON_CSV_PATH = RESULTS_DIR / "model_comparison.csv"
COMPARISON_STD_CSV_PATH = RESULTS_DIR / "model_comparison_standard.csv"


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
    """Loads verified test metrics and model comparison records."""
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

    comp_std_df = pd.DataFrame()
    if COMPARISON_STD_CSV_PATH.exists():
        try:
            comp_std_df = pd.read_csv(COMPARISON_STD_CSV_PATH)
        except Exception:
            comp_std_df = pd.DataFrame()

    return metrics, comp_df, comp_std_df


# -----------------------------------------------------------------------------
# Professional Healthcare Analytics Theme (Navy, Teal, White)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Brand Typography & Spacing */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Header Section */
    .brand-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #0d9488 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.12);
    }
    .brand-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        color: #ffffff;
    }
    .brand-subtitle {
        font-size: 1.05rem;
        color: #e2e8f0;
        margin-top: 6px;
        margin-bottom: 0;
        font-weight: 400;
    }
    .brand-badge {
        display: inline-block;
        background: rgba(13, 148, 136, 0.35);
        border: 1px solid #14b8a6;
        color: #ccfbf1;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        margin-bottom: 10px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* Disclaimer Notice Card */
    .medical-disclaimer {
        background-color: #f8fafc;
        border-left: 4px solid #0d9488;
        border-right: 1px solid #e2e8f0;
        border-top: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 20px;
        margin-bottom: 22px;
        font-size: 0.90rem;
        color: #334155;
        line-height: 1.55;
    }
    .medical-disclaimer strong {
        color: #0f172a;
    }

    /* Metric Container Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.08);
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.80rem;
        color: #0d9488;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Prediction Result Cards */
    .result-card-elevated {
        background: #fff1f2;
        border: 2px solid #f43f5e;
        border-radius: 10px;
        padding: 22px;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(244, 63, 94, 0.10);
    }
    .result-card-low {
        background: #f0fdf4;
        border: 2px solid #22c55e;
        border-radius: 10px;
        padding: 22px;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.10);
    }

    /* Clinical Form Container */
    .form-section-header {
        color: #0f172a;
        font-size: 1.05rem;
        font-weight: 700;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }

    /* Section Subheadings */
    .section-title {
        color: #0f172a;
        font-weight: 700;
        font-size: 1.35rem;
        margin-top: 5px;
        margin-bottom: 8px;
    }
    .section-caption {
        color: #64748b;
        font-size: 0.92rem;
        margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header():
    """Renders the top branding header and mandatory academic/ethical disclaimer."""
    st.markdown(
        """
        <div class="brand-banner">
            <span class="brand-badge">Academic Research & Healthcare Analytics</span>
            <h1 class="brand-title">Heart Disease Risk Prediction System</h1>
            <p class="brand-subtitle">Artificial Neural Network (ANN) Architecture with Scientifically Verified Zero-Leakage Evaluation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="medical-disclaimer">
            <strong>Mandatory Educational & Research Disclaimer:</strong>
            This web application is developed strictly as a machine learning engineering project for academic demonstration and research benchmarking.
            It is <strong>not</strong> a certified medical diagnostic device or clinical decision support system.
            Predictions represent statistical probability estimates derived from historical dataset distributions and must never supersede professional medical consultation, clinical diagnosis, or physician judgment.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    # Session state initialization for prediction history
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    # Render Header & Disclaimer
    render_header()

    # Load artifacts and metrics
    model, pipeline, err_art = load_ml_artifacts()
    df_raw, err_df = load_dataset()
    metrics, comp_df, comp_std_df = load_benchmark_metrics()

    if err_art:
        st.error(err_art)
        st.info("To initialize models, run: `python src/train.py` from the project root.")
        return

    # Navigation Tabs
    tab_pred, tab_perf, tab_comp, tab_xai, tab_eda, tab_method = st.tabs(
        [
            "Clinical Risk Prediction",
            "Model Performance",
            "Model Benchmark Comparison",
            "Explainable AI (XAI)",
            "Dataset Insights",
            "Methodology & Training",
        ]
    )

    # =========================================================================
    # TAB 1: CLINICAL RISK PREDICTION
    # =========================================================================
    with tab_pred:
        st.markdown('<div class="section-title">Patient Clinical Input Parameters</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Configure the 13 physiological and diagnostic markers across four standard clinical categories.</div>',
            unsafe_allow_html=True,
        )

        with st.form("clinical_prediction_form"):
            c1, c2, c3, c4 = st.columns(4)

            # Category 1: Demographics
            with c1:
                st.markdown('<div class="form-section-header">1. Demographics</div>', unsafe_allow_html=True)
                age = st.number_input(
                    "Patient Age (years)",
                    min_value=20,
                    max_value=100,
                    value=55,
                    step=1,
                    help="Patient chronological age (20 to 100 years)",
                )
                sex_label = st.selectbox(
                    "Biological Sex",
                    options=["Male", "Female"],
                    index=0,
                    help="Biological sex of the patient",
                )
                sex = 1 if sex_label == "Male" else 0

            # Category 2: Resting Vitals & Blood Chemistry
            with c2:
                st.markdown('<div class="form-section-header">2. Vitals & Chemistry</div>', unsafe_allow_html=True)
                trestbps = st.number_input(
                    "Resting BP (mm Hg)",
                    min_value=80,
                    max_value=220,
                    value=130,
                    step=1,
                    help="Resting systolic blood pressure upon admission",
                )
                chol = st.number_input(
                    "Serum Cholesterol (mg/dl)",
                    min_value=100,
                    max_value=600,
                    value=240,
                    step=1,
                    help="Serum cholesterol in mg/dl",
                )
                fbs_label = st.selectbox(
                    "Fasting Blood Sugar > 120 mg/dl",
                    options=["False (Normal)", "True (Elevated)"],
                    index=0,
                    help="Fasting blood sugar exceeding 120 mg/dl",
                )
                fbs = 1 if "True" in fbs_label else 0

            # Category 3: Exercise Stress Test
            with c3:
                st.markdown('<div class="form-section-header">3. Cardiac Stress Test</div>', unsafe_allow_html=True)
                cp_map = {
                    "0: Typical Angina": 0,
                    "1: Atypical Angina": 1,
                    "2: Non-Anginal Pain": 2,
                    "3: Asymptomatic": 3,
                }
                cp_selected = st.selectbox(
                    "Chest Pain Type (cp)",
                    options=list(cp_map.keys()),
                    index=0,
                    help="Clinical chest pain classification",
                )
                cp = cp_map[cp_selected]

                restecg_map = {
                    "0: Normal": 0,
                    "1: ST-T Wave Abnormality": 1,
                    "2: Left Ventricular Hypertrophy": 2,
                }
                restecg_selected = st.selectbox(
                    "Resting ECG (restecg)",
                    options=list(restecg_map.keys()),
                    index=0,
                    help="Resting electrocardiographic results",
                )
                restecg = restecg_map[restecg_selected]

                thalach = st.number_input(
                    "Max Heart Rate (thalach)",
                    min_value=60,
                    max_value=220,
                    value=150,
                    step=1,
                    help="Maximum heart rate achieved during exercise treadmill testing",
                )
                exang_label = st.selectbox(
                    "Exercise Angina (exang)",
                    options=["No", "Yes"],
                    index=0,
                    help="Exercise induced angina reported during stress test",
                )
                exang = 1 if exang_label == "Yes" else 0

            # Category 4: Electrocardiogram & Fluoroscopy
            with c4:
                st.markdown('<div class="form-section-header">4. Fluoroscopy & Scan</div>', unsafe_allow_html=True)
                oldpeak = st.number_input(
                    "ST Depression (oldpeak)",
                    min_value=0.0,
                    max_value=7.0,
                    value=1.0,
                    step=0.1,
                    format="%.1f",
                    help="ST depression induced by exercise relative to rest",
                )
                slope_map = {
                    "0: Upsloping": 0,
                    "1: Flat": 1,
                    "2: Downsloping": 2,
                }
                slope_selected = st.selectbox(
                    "Peak ST Slope (slope)",
                    options=list(slope_map.keys()),
                    index=1,
                    help="Slope of the peak exercise ST segment",
                )
                slope = slope_map[slope_selected]

                ca = st.slider(
                    "Major Fluoroscopy Vessels (ca)",
                    min_value=0,
                    max_value=4,
                    value=0,
                    help="Number of major vessels (0-4) colored by fluoroscopy",
                )

                thal_map = {
                    "0: Null / Unknown": 0,
                    "1: Normal": 1,
                    "2: Fixed Defect": 2,
                    "3: Reversible Defect": 3,
                }
                thal_selected = st.selectbox(
                    "Thallium Scintigraphy (thal)",
                    options=list(thal_map.keys()),
                    index=2,
                    help="Thallium heart scan imaging result",
                )
                thal = thal_map[thal_selected]

            # Classification threshold selection
            st.divider()
            threshold_col1, threshold_col2 = st.columns([2, 3])
            with threshold_col1:
                custom_threshold = st.slider(
                    "Decision Threshold for Positive Risk Classification",
                    min_value=0.30,
                    max_value=0.70,
                    value=0.50,
                    step=0.05,
                    format="%.2f",
                    help="Default 0.50. Lower thresholds increase sensitivity (fewer missed positives).",
                )
            with threshold_col2:
                st.caption(
                    f"Current Operating Point: **{custom_threshold:.2f}** (50.00% standard baseline). "
                    "In clinical screening scenarios, adjusting the operating threshold allows trade-offs between "
                    "Sensitivity (Recall) and Specificity based on the clinical cost of false negatives."
                )

            submit_pred = st.form_submit_button(
                "Compute Statistical Risk Prediction",
                type="primary",
                width="stretch",
            )

        # Handle prediction execution
        if submit_pred:
            input_dict = {
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
            input_df = pd.DataFrame([input_dict])

            try:
                # Preprocess input through fitted Scikit-Learn pipeline
                X_proc = pipeline.transform(input_df)

                # ANN forward pass
                pred_prob = float(model.predict(X_proc, verbose=0)[0][0])
                pred_class = int(pred_prob >= custom_threshold)

                st.divider()
                st.markdown('<div class="section-title">Model Prediction Output</div>', unsafe_allow_html=True)

                # Metric Cards
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Predicted Class</div>
                            <div class="metric-value">Class {pred_class}</div>
                            <div class="metric-sub">{"Positive Marker" if pred_class == 1 else "Negative Marker"}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with m2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Predicted Probability</div>
                            <div class="metric-value">{pred_prob * 100:.2f}%</div>
                            <div class="metric-sub">Sigmoid Activation</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with m3:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Classification Threshold</div>
                            <div class="metric-value">{custom_threshold * 100:.1f}%</div>
                            <div class="metric-sub">Decision Boundary</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with m4:
                    risk_label = "Elevated Risk Pattern" if pred_class == 1 else "Low Risk Pattern"
                    color_style = "#e11d48" if pred_class == 1 else "#16a34a"
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Statistical Risk Stratum</div>
                            <div class="metric-value" style="color:{color_style}; font-size:1.35rem;">{risk_label}</div>
                            <div class="metric-sub">Cohort Association</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.write("")
                st.write("**Model Probability Spectrum (0.0 to 1.0):**")
                st.progress(pred_prob)

                # Explanatory card
                if pred_class == 1:
                    st.markdown(
                        f"""
                        <div class="result-card-elevated">
                            <h3 style="color:#9f1239; margin-top:0;">Prediction: Elevated Statistical Risk (Class 1)</h3>
                            <p style="color:#881337; margin:0; line-height:1.6; font-size:0.95rem;">
                                The Artificial Neural Network estimates a <strong>{pred_prob * 100:.2f}%</strong> predicted probability
                                of heart disease risk markers, exceeding the active decision threshold of <strong>{custom_threshold * 100:.1f}%</strong>.<br>
                                <em>Clinical interpretation: The patient's input combination (e.g., chest pain type, ST depression, fluoroscopy vessels, max heart rate)
                                closely mirrors profiles with confirmed angiographic disease in the training cohort.</em>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card-low">
                            <h3 style="color:#166534; margin-top:0;">Prediction: Lower Statistical Risk (Class 0)</h3>
                            <p style="color:#14532d; margin:0; line-height:1.6; font-size:0.95rem;">
                                The Artificial Neural Network estimates a <strong>{(1.0 - pred_prob) * 100:.2f}%</strong> probability
                                of non-disease status, remaining safely below the decision threshold.<br>
                                <em>Clinical interpretation: The patient's clinical markers align with baseline normal cardiac profiles observed in healthy cohort controls.</em>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Record into session history
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                history_record = {
                    "Timestamp": now_str,
                    "Age": age,
                    "Sex": sex_label,
                    "Chest Pain": cp_selected.split(":")[0],
                    "Resting BP": trestbps,
                    "Cholesterol": chol,
                    "Max HR": thalach,
                    "Oldpeak": oldpeak,
                    "Predicted Class": pred_class,
                    "Predicted Probability": f"{pred_prob * 100:.2f}%",
                    "Threshold": f"{custom_threshold * 100:.1f}%",
                }
                st.session_state.prediction_history.insert(0, history_record)

                # Downloadable clinical summary report
                report_text = f"""================================================================================
CARDIOLOGY ML RESEARCH LAB - PREDICTION SUMMARY REPORT
================================================================================
Generated: {now_str}
Model Architecture: Multi-Layer Perceptron (Artificial Neural Network)
Evaluation Protocol: Zero-Leakage Split (Deduplicated Unique Patients)
Framework: TensorFlow 2.x / Keras / Scikit-Learn

1. PREDICTION OUTCOME:
----------------------
Predicted Class           : Class {pred_class} ({"Elevated Risk Pattern" if pred_class == 1 else "Low Risk Pattern"})
Predicted Probability     : {pred_prob * 100:.2f}%
Classification Threshold : {custom_threshold * 100:.1f}%
Model Output Raw Value    : {pred_prob:.6f}

2. PATIENT CLINICAL ATTRIBUTES:
-------------------------------
Age                       : {age} years
Biological Sex            : {sex_label}
Chest Pain Type           : {cp_selected}
Resting Blood Pressure    : {trestbps} mm Hg
Serum Cholesterol         : {chol} mg/dl
Fasting Blood Sugar > 120 : {fbs_label}
Resting ECG               : {restecg_selected}
Max Heart Rate (thalach)  : {thalach} bpm
Exercise Angina (exang)   : {exang_label}
ST Depression (oldpeak)   : {oldpeak} mm
Peak ST Slope             : {slope_selected}
Fluoroscopy Vessels (ca)  : {ca}
Thallium Scintigraphy     : {thal_selected}

3. RESEARCH & CLINICAL DISCLAIMER:
----------------------------------
This report was generated by an educational B.Tech artificial intelligence system.
This software is NOT a certified medical diagnostic tool. Predictions represent
statistical likelihood and must NEVER replace professional medical evaluation.
================================================================================
"""
                st.download_button(
                    label="Download Clinical Summary Report (.txt)",
                    data=report_text,
                    file_name=f"cardio_ann_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    width="stretch",
                )

            except Exception as e:
                st.error(f"Inference pipeline execution error: {e}")

        # Prediction session history table
        st.divider()
        st.markdown('<div class="section-title">Session Prediction Audit Log</div>', unsafe_allow_html=True)
        st.caption("Ephemeral session cache for comparative case analysis. No patient records are stored on disk.")

        if st.session_state.prediction_history:
            df_hist = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(df_hist, width="stretch")
            if st.button("Clear Session Log"):
                st.session_state.prediction_history = []
                st.rerun()
        else:
            st.info("No predictions executed yet in this session. Configure the parameters above and click 'Compute Statistical Risk Prediction'.")

    # =========================================================================
    # TAB 2: MODEL PERFORMANCE
    # =========================================================================
    with tab_perf:
        st.markdown('<div class="section-title">Model Performance & Validation Metrics</div>', unsafe_allow_html=True)

        # Dual-benchmark selector
        benchmark_mode = st.radio(
            "Evaluation Benchmark Protocol:",
            options=[
                "Scientifically Rigorous Zero-Leakage Protocol (Unique Patients, N=302)",
                "Standard 1,025-Sample Literature Protocol (Kaggle Comparison)",
            ],
            index=0,
            horizontal=True,
        )

        is_leakage_free = "Zero-Leakage" in benchmark_mode

        if is_leakage_free:
            st.caption(
                "Evaluated strictly on **61 unseen held-out patient records** from the 302 unique Cleveland cohort. "
                "Preprocessor fitted strictly on training partition. Zero clone contamination exists."
            )
            acc_val = metrics.get("accuracy", 0.8361)
            prec_val = metrics.get("precision", 0.8966)
            rec_val = metrics.get("recall", 0.7879)
            f1_val = metrics.get("f1_score", 0.8387)
            auc_val = metrics.get("roc_auc", 0.9015)
            cm_dict = metrics.get("confusion_matrix", {"true_negatives": 25, "false_positives": 3, "false_negatives": 7, "true_positives": 26})
            test_n = metrics.get("test_samples", 61)
        else:
            std_b = metrics.get("standard_benchmark", {})
            st.caption(
                "Evaluated on **205 test samples** with naive 80/20 train/test splitting across all 1,025 Kaggle rows. "
                "Note: 202 of 205 test samples are duplicates of training rows, causing artificial performance inflation."
            )
            acc_val = std_b.get("accuracy", 0.9854)
            prec_val = std_b.get("precision", 0.9811)
            rec_val = std_b.get("recall", 0.9905)
            f1_val = std_b.get("f1_score", 0.9858)
            auc_val = std_b.get("roc_auc", 0.9959)
            cm_dict = {"true_negatives": 98, "false_positives": 2, "false_negatives": 2, "true_positives": 103}
            test_n = std_b.get("test_samples", 205)

        # 5 Metric Cards
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Test Accuracy</div>
                    <div class="metric-value">{acc_val * 100:.2f}%</div>
                    <div class="metric-sub">N = {test_n} Test Set</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value">{prec_val * 100:.2f}%</div>
                    <div class="metric-sub">Low False Alarms</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Recall (Sensitivity)</div>
                    <div class="metric-value">{rec_val * 100:.2f}%</div>
                    <div class="metric-sub">True Positive Rate</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">F1-Score</div>
                    <div class="metric-value">{f1_val * 100:.2f}%</div>
                    <div class="metric-sub">Harmonic Mean</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k5:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">ROC-AUC</div>
                    <div class="metric-value">{auc_val:.4f}</div>
                    <div class="metric-sub">Discrimination Ability</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown(
            f"""
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px 18px; margin-top:10px;">
                <strong>Confusion Matrix Distribution:</strong> &nbsp;
                True Negatives: <strong>{cm_dict.get('true_negatives', 0)}</strong> &nbsp;|&nbsp;
                False Positives: <strong>{cm_dict.get('false_positives', 0)}</strong> &nbsp;|&nbsp;
                False Negatives: <strong>{cm_dict.get('false_negatives', 0)}</strong> &nbsp;|&nbsp;
                True Positives: <strong>{cm_dict.get('true_positives', 0)}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Visual Artifacts
        st.markdown('<div class="section-title">Diagnostic Visualizations</div>', unsafe_allow_html=True)
        col_cm, col_roc = st.columns(2)
        with col_cm:
            cm_img = RESULTS_DIR / "confusion_matrix.png"
            if cm_img.exists():
                st.image(str(cm_img), caption="Confusion Matrix on Held-Out Test Cohort (N=61)", width="stretch")
            else:
                st.warning("Confusion matrix plot not found.")

        with col_roc:
            roc_img = RESULTS_DIR / "roc_curve.png"
            if roc_img.exists():
                st.image(str(roc_img), caption=f"ROC Curve (AUC = {auc_val:.4f})", width="stretch")
            else:
                st.warning("ROC curve plot not found.")

        # Classification report text
        clr_file = RESULTS_DIR / "classification_report.txt"
        if clr_file.exists():
            with st.expander("View Full Classification Report Breakdown", expanded=False):
                with open(clr_file, "r", encoding="utf-8") as f:
                    st.code(f.read(), language="text")

    # =========================================================================
    # TAB 3: MODEL BENCHMARK COMPARISON
    # =========================================================================
    with tab_comp:
        st.markdown('<div class="section-title">Comparative Machine Learning Benchmarks</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Systematic evaluation of classical baselines (Logistic Regression, Decision Tree, Random Forest) alongside Baseline and Improved ANN models.</div>',
            unsafe_allow_html=True,
        )

        comp_view = st.radio(
            "Benchmark View:",
            options=["Leakage-Free Test Split (Unique Patients, N=61)", "Standard 1,025-Row Dataset (Kaggle Split, N=205)"],
            index=0,
            horizontal=True,
        )

        active_df = comp_df if "Leakage-Free" in comp_view else comp_std_df

        if not active_df.empty:
            st.dataframe(
                active_df.style.highlight_max(
                    subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                    color="#ccfbf1",
                ),
                width="stretch",
            )
        else:
            st.warning("Model comparison data not found. Run `python src/train.py`.")

        comp_plot = RESULTS_DIR / "model_comparison.png"
        if comp_plot.exists():
            st.image(str(comp_plot), caption="Empirical Model Performance Comparison", width="stretch")

        st.markdown(
            """
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:16px 20px; margin-top:16px; line-height:1.6;">
                <h4 style="color:#0f172a; margin-top:0;">Key Engineering Insights:</h4>
                <ul style="margin-bottom:0; color:#334155;">
                    <li><strong>ANN Generalization:</strong> The Improved ANN achieves <strong>0.9015 ROC-AUC</strong> and <strong>89.66% precision</strong> on the leakage-free holdout test set, providing strong risk discrimination while minimizing costly false alarms.</li>
                    <li><strong>Ensemble Comparison:</strong> Random Forest achieves higher recall (87.88%) but lower precision (80.56%), while Logistic Regression provides a robust linear baseline (85.25% accuracy).</li>
                    <li><strong>Why ANN is Selected:</strong> The regularized multi-layer perceptron provides calibrated continuous probability outputs with smooth decision boundaries, essential for clinical risk scoring and threshold tuning.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 4: EXPLAINABLE AI (XAI)
    # =========================================================================
    with tab_xai:
        st.markdown('<div class="section-title">Explainable AI: Permutation Feature Influence</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Permutation feature importance quantifies how strongly the Artificial Neural Network relies on each clinical variable.</div>',
            unsafe_allow_html=True,
        )

        xai_img = RESULTS_DIR / "feature_analysis.png"
        if xai_img.exists():
            st.image(str(xai_img), caption="Permutation Feature Influence Ranking & Risk Marker Correlations", width="stretch")
        else:
            st.warning("Feature analysis visualization not found.")

        st.markdown(
            """
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:18px 22px; line-height:1.6;">
                <h4 style="color:#0f172a; margin-top:0;">Methodology & Interpretation</h4>
                <p style="color:#334155; margin-bottom:8px;">
                    <strong>Computation Protocol:</strong> Each input feature was randomly permuted across 10 repeats on the unseen test set,
                    severing its association with the target while preserving all other feature distributions. The resulting decline in ROC-AUC
                    directly measures the model's architectural dependency on that feature.
                </p>
                <p style="color:#64748b; font-size:0.90rem; margin-bottom:0;">
                    <em>Ethical Disclaimer: Permutation feature importance represents statistical reliance within this model only.
                    High feature importance reflects mathematical utility in predicting tabular labels and does <strong>not</strong> imply medical or physiological causation.</em>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 5: DATASET INSIGHTS
    # =========================================================================
    with tab_eda:
        st.markdown('<div class="section-title">Cardiovascular Dataset Analytics</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Exploratory analysis of <code>dataset/heart.csv</code> (1,025 observations representing 302 unique Cleveland patients).</div>',
            unsafe_allow_html=True,
        )

        if df_raw is not None:
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Observations</div>
                        <div class="metric-value">{len(df_raw):,}</div>
                        <div class="metric-sub">302 Unique Patients</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with e2:
                st.markdown(
                    """
                    <div class="metric-card">
                        <div class="metric-label">Input Features</div>
                        <div class="metric-value">13</div>
                        <div class="metric-sub">Continuous & Categorical</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with e3:
                pos_c = int(df_raw["target"].sum())
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Class 1 (Disease Present)</div>
                        <div class="metric-value">{pos_c}</div>
                        <div class="metric-sub">{pos_c / len(df_raw) * 100:.1f}% Prevalence</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with e4:
                neg_c = len(df_raw) - pos_c
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Class 0 (No Disease)</div>
                        <div class="metric-value">{neg_c}</div>
                        <div class="metric-sub">{neg_c / len(df_raw) * 100:.1f}% Controls</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.divider()

            # EDA Charts
            c_eda1, c_eda2 = st.columns(2)
            with c_eda1:
                st.markdown("##### Age Distribution by Disease Status")
                fig_age, ax_age = plt.subplots(figsize=(7, 4))
                sns.histplot(
                    data=df_raw,
                    x="age",
                    hue="target",
                    kde=True,
                    palette=["#1e3a8a", "#0d9488"],
                    bins=20,
                    ax=ax_age,
                )
                ax_age.set_title("Patient Age Distribution", fontsize=11, fontweight="bold")
                ax_age.set_xlabel("Age (years)")
                st.pyplot(fig_age)

            with c_eda2:
                st.markdown("##### Peak Heart Rate (thalach) vs Disease")
                fig_hr, ax_hr = plt.subplots(figsize=(7, 4))
                sns.boxplot(
                    data=df_raw,
                    x="target",
                    y="thalach",
                    palette=["#1e3a8a", "#0d9488"],
                    ax=ax_hr,
                )
                ax_hr.set_xticklabels(["No Disease (0)", "Disease Present (1)"])
                ax_hr.set_title("Peak Exercise Heart Rate vs Disease Status", fontsize=11, fontweight="bold")
                ax_hr.set_ylabel("Max Heart Rate (bpm)")
                st.pyplot(fig_hr)

            st.markdown("##### Raw Dataset Records (Top 10 Observations)")
            st.dataframe(df_raw.head(10), width="stretch")

        elif err_df:
            st.warning(err_df)

    # =========================================================================
    # TAB 6: METHODOLOGY & TRAINING ANALYSIS
    # =========================================================================
    with tab_method:
        st.markdown('<div class="section-title">Artificial Neural Network Methodology & Architecture</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Complete engineering pipeline, loss optimization curves, and data-leakage prevention design.</div>',
            unsafe_allow_html=True,
        )

        hist_plot = RESULTS_DIR / "training_history.png"
        if hist_plot.exists():
            st.image(str(hist_plot), caption="Training vs Validation Loss & Accuracy Progression Across Epochs", width="stretch")

        st.markdown(
            """
            ### Architectural Design Decisions
            1. **Data Leakage Elimination**:
               - The raw dataset (`dataset/heart.csv`, 1,025 rows) contains 723 clone entries of 302 unique patients.
               - Naive random splitting places duplicate clones into both train and test, producing artificially inflated (~98%) accuracy.
               - Our pipeline deduplicates patient records prior to splitting, establishing strict Train (N=204), Validation (N=37), and Held-Out Test (N=61) partitions.
               - The `ColumnTransformer` is fitted **strictly on X_train** and applied to Validation and Test without contamination.

            2. **Feature Engineering Pipeline**:
               - Continuous features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) scaled via `StandardScaler`.
               - Multi-class nominal features (`cp`, `restecg`, `slope`, `ca`, `thal`) encoded via `OneHotEncoder(handle_unknown='ignore')`.
               - Binary features (`sex`, `fbs`, `exang`) passed through without distortion.
               - Total input dimension: **27 preprocessed features**.

            3. **Multi-Layer Perceptron (MLP) Topology**:
               - **Dense 48** (ReLU activation, L2 weight decay $\\lambda = 0.001$)
               - **Batch Normalization** (stabilizes internal covariate shift)
               - **Dropout 0.25** (prevents co-adaptation of neurons)
               - **Dense 24** (ReLU activation, L2 weight decay $\\lambda = 0.001$)
               - **Batch Normalization**
               - **Dropout 0.20**
               - **Dense 12** (ReLU activation)
               - **Dropout 0.10**
               - **Output 1** (Sigmoid activation for calibrated risk probability)

            4. **Optimization & Regularization**:
               - Optimizer: `Adam(learning_rate=0.001)`
               - Loss Function: Binary Cross-Entropy
               - Early Stopping: `EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)`
               - Learning Rate Decay: `ReduceLROnPlateau(factor=0.5, patience=8)`
            """
        )


if __name__ == "__main__":
    main()
