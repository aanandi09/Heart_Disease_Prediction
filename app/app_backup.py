"""
Streamlit Web Application: Heart Disease Prediction using ANN
==============================================================
A comprehensive, academically rigorous, and resume-worthy B.Tech AI/ML dashboard.
Features:
1. 🔮 Clinical Prediction (Interactive ANN inference with full 13 clinical markers)
2. 📈 Model Performance & Benchmarking (Real evaluation metrics & comparative models)
3. 📊 Dataset Insights & EDA (Interactive distributions, correlations, and patient cohorts)
4. 🧠 Model Explainability (Permutation feature importance & risk factor attribution)
5. ℹ️ Project Methodology & Documentation (Academic architecture & ethical disclaimer)
"""

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
    page_title="Heart Disease Prediction | ANN Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Relative directory references
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "heart.csv"
MODEL_PATH = BASE_DIR / "models" / "heart_disease_ann.keras"
PIPELINE_PATH = BASE_DIR / "models" / "preprocessing_pipeline.pkl"
RESULTS_DIR = BASE_DIR / "results"
METRICS_JSON_PATH = RESULTS_DIR / "final_metrics.json"
COMPARISON_CSV_PATH = RESULTS_DIR / "model_comparison.csv"


# -----------------------------------------------------------------------------
# Cached Data & Artifact Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ml_artifacts():
    """Loads and caches the trained Keras ANN model and Scikit-Learn pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Pipeline file not found at: {PIPELINE_PATH}")

    model = keras.models.load_model(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    return model, pipeline


@st.cache_data
def load_dataset():
    """Loads and caches the raw heart disease dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


@st.cache_data
def load_benchmark_metrics():
    """Loads final test metrics and comparison benchmark data."""
    final_metrics = {}
    if METRICS_JSON_PATH.exists():
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            final_metrics = json.load(f)

    comparison_df = pd.DataFrame()
    if COMPARISON_CSV_PATH.exists():
        comparison_df = pd.read_csv(COMPARISON_CSV_PATH)

    return final_metrics, comparison_df


# -----------------------------------------------------------------------------
# Custom Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        font-size: 1.02rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .disclaimer-card {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 12px 18px;
        border-radius: 8px;
        font-size: 0.92rem;
        color: #92400e;
        margin-bottom: 1.5rem;
    }
    .prediction-card-high {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-top: 20px;
    }
    .prediction-card-low {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        margin-top: 20px;
    }
    .metric-badge {
        background: #f3f4f6;
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
def main():
    # Header Banner
    st.markdown('<div class="main-title">❤️ Heart Disease Prediction using Artificial Neural Network</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">B.Tech CSE (AI/ML) Academic Project — Deep Multi-Layer Perceptron (ANN) with Benchmark Comparisons & Explainability</div>',
        unsafe_allow_html=True,
    )

    # Mandatory Medical & Educational Disclaimer
    st.markdown(
        """
        <div class="disclaimer-card">
            <strong>⚠️ Medical & Educational Disclaimer:</strong><br>
            This software is developed strictly for educational, academic demonstration, and research purposes.
            It is <strong>NOT</strong> a certified medical diagnostic system. Model predictions should never be used as a substitute
            for professional clinical judgment, physician evaluation, or healthcare diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load ML Artifacts
    try:
        model, pipeline = load_ml_artifacts()
        df = load_dataset()
        metrics, comp_df = load_benchmark_metrics()
    except Exception as err:
        st.error(f"Error loading system artifacts: {err}")
        st.info("Run `python src/train.py` and `python src/evaluate.py` to regenerate all model artifacts.")
        return

    # Navigation Tabs
    tab_pred, tab_perf, tab_data, tab_xai, tab_about = st.tabs(
        [
            "🔮 Clinical Prediction",
            "📈 Model Performance & Benchmarks",
            "📊 Dataset Insights & EDA",
            "🧠 Model Explainability",
            "ℹ️ Project Methodology",
        ]
    )

    # =========================================================================
    # TAB 1: CLINICAL PREDICTION
    # =========================================================================
    with tab_pred:
        st.subheader("📋 Enter Patient Clinical Parameters")
        st.write("Provide the 13 diagnostic indicators below to compute the neural network risk probability.")

        with st.form("clinical_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("##### 👤 Demographic & Symptoms")
                age = st.number_input(
                    "1. Age (in years)",
                    min_value=20,
                    max_value=110,
                    value=55,
                    step=1,
                    help="Patient age at examination",
                )

                sex_display = st.selectbox(
                    "2. Sex",
                    options=["Male", "Female"],
                    index=0,
                    help="Biological sex of the patient",
                )
                sex = 1 if sex_display == "Male" else 0

                cp_dict = {
                    "Typical Angina (0)": 0,
                    "Atypical Angina (1)": 1,
                    "Non-anginal Pain (2)": 2,
                    "Asymptomatic (3)": 3,
                }
                cp_sel = st.selectbox(
                    "3. Chest Pain Type (cp)",
                    options=list(cp_dict.keys()),
                    index=0,
                    help="Type 0: Typical Angina | Type 1: Atypical Angina | Type 2: Non-anginal | Type 3: Asymptomatic",
                )
                cp = cp_dict[cp_sel]

                exang_disp = st.selectbox(
                    "4. Exercise-Induced Angina (exang)",
                    options=["No (0)", "Yes (1)"],
                    index=0,
                    help="Whether exercise induces angina pain",
                )
                exang = 1 if "Yes" in exang_disp else 0

            with col2:
                st.markdown("##### 🩺 Cardiovascular Vitals")
                trestbps = st.number_input(
                    "5. Resting Blood Pressure (trestbps in mm Hg)",
                    min_value=80,
                    max_value=230,
                    value=130,
                    step=1,
                    help="Resting blood pressure on hospital admission",
                )

                chol = st.number_input(
                    "6. Serum Cholesterol (chol in mg/dl)",
                    min_value=100,
                    max_value=600,
                    value=240,
                    step=1,
                    help="Serum cholesterol level",
                )

                thalach = st.number_input(
                    "7. Maximum Heart Rate (thalach in bpm)",
                    min_value=60,
                    max_value=225,
                    value=150,
                    step=1,
                    help="Peak heart rate achieved during exercise stress test",
                )

                fbs_disp = st.selectbox(
                    "8. Fasting Blood Sugar > 120 mg/dl (fbs)",
                    options=["No / Normal (<= 120 mg/dl)", "Yes / Elevated (> 120 mg/dl)"],
                    index=0,
                    help="Fasting blood sugar above 120 mg/dl",
                )
                fbs = 1 if "Yes" in fbs_disp else 0

            with col3:
                st.markdown("##### 📉 Diagnostic & ECG Markers")
                restecg_dict = {
                    "Normal (0)": 0,
                    "ST-T Wave Abnormality (1)": 1,
                    "Left Ventricular Hypertrophy (2)": 2,
                }
                restecg_sel = st.selectbox(
                    "9. Resting ECG Results (restecg)",
                    options=list(restecg_dict.keys()),
                    index=0,
                    help="Resting electrocardiogram results",
                )
                restecg = restecg_dict[restecg_sel]

                oldpeak = st.number_input(
                    "10. ST Depression (oldpeak)",
                    min_value=0.0,
                    max_value=7.0,
                    value=1.0,
                    step=0.1,
                    format="%.1f",
                    help="ST depression induced by exercise relative to rest",
                )

                slope_dict = {
                    "Upsloping (0)": 0,
                    "Flat (1)": 1,
                    "Downsloping (2)": 2,
                }
                slope_sel = st.selectbox(
                    "11. Peak Exercise ST Slope (slope)",
                    options=list(slope_dict.keys()),
                    index=1,
                    help="Slope of the peak exercise ST segment",
                )
                slope = slope_dict[slope_sel]

                ca = st.selectbox(
                    "12. Number of Major Vessels Colored (ca)",
                    options=[0, 1, 2, 3, 4],
                    index=0,
                    help="Number of major blood vessels (0-4) colored by fluoroscopy",
                )

                thal_dict = {
                    "Unknown / Null (0)": 0,
                    "Normal Blood Flow (1)": 1,
                    "Fixed Defect (2)": 2,
                    "Reversible Defect (3)": 3,
                }
                thal_sel = st.selectbox(
                    "13. Thalassemia Category (thal)",
                    options=list(thal_dict.keys()),
                    index=2,
                    help="Thalassemia cardiac blood flow disorder status",
                )
                thal = thal_dict[thal_sel]

            submit = st.form_submit_button("🔮 Predict Heart Disease Likelihood", use_container_width=True)

        if submit:
            # Build input DataFrame
            input_df = pd.DataFrame(
                [
                    {
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
                ]
            )

            st.divider()
            st.subheader("📊 Neural Network Inference Result")

            try:
                # Transform using saved pipeline
                X_proc = pipeline.transform(input_df)

                # Predict probability with ANN
                raw_prob = float(model.predict(X_proc, verbose=0)[0][0])
                predicted_class = int(raw_prob >= 0.5)

                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric(
                        label="Predicted Classification",
                        value=f"Class {predicted_class}",
                        delta="High Likelihood" if predicted_class == 1 else "Low Likelihood",
                        delta_color="inverse" if predicted_class == 1 else "normal",
                    )
                with m2:
                    st.metric(
                        label="Model-Estimated Probability",
                        value=f"{raw_prob * 100:.2f}%",
                    )
                with m3:
                    st.metric(
                        label="Decision Threshold",
                        value="50.00%",
                    )
                with m4:
                    st.metric(
                        label="Risk Assessment",
                        value="Elevated Risk" if predicted_class == 1 else "Low Risk",
                    )

                # Confidence Gauge
                st.write("**Model Output Probability Gauge:**")
                st.progress(raw_prob)

                # Objective result card with required language
                if predicted_class == 1:
                    st.markdown(
                        f"""
                        <div class="prediction-card-high">
                            <h3 style="color: #dc2626; margin: 0;">Prediction: Higher predicted likelihood of heart disease (Class 1)</h3>
                            <p style="color: #7f1d1d; margin-top: 10px; font-size: 1.05rem;">
                                Predicted Class 1 based on the learned patterns in the dataset.<br>
                                The model predicts a higher likelihood for Class 1 with an estimated probability of <strong>{raw_prob * 100:.2f}%</strong>.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="prediction-card-low">
                            <h3 style="color: #16a34a; margin: 0;">Prediction: Lower predicted likelihood of heart disease (Class 0)</h3>
                            <p style="color: #14532d; margin-top: 10px; font-size: 1.05rem;">
                                Predicted Class 0 based on the learned patterns in the dataset.<br>
                                The model predicts a lower likelihood for Class 1 (Estimated non-disease probability: <strong>{(1.0 - raw_prob) * 100:.2f}%</strong>).
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with st.expander("🔍 View Normalized Input Features", expanded=False):
                    st.dataframe(input_df, use_container_width=True)

            except Exception as e:
                st.error(f"Inference error: {e}")

    # =========================================================================
    # TAB 2: MODEL PERFORMANCE & BENCHMARKS
    # =========================================================================
    with tab_perf:
        st.subheader("📈 Final ANN Model Performance on Unseen Test Set")
        st.caption("Evaluated on 205 held-out patient samples with stratified 80/20 train/test splitting.")

        # Metric Badges
        if metrics:
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Test Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f}%")
            k2.metric("Precision", f"{metrics.get('precision', 0) * 100:.2f}%")
            k3.metric("Recall (Sensitivity)", f"{metrics.get('recall', 0) * 100:.2f}%")
            k4.metric("F1-Score", f"{metrics.get('f1_score', 0) * 100:.2f}%")
            k5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")

        st.divider()

        # Model Comparison Table
        st.subheader("🏆 Model Benchmark Comparison")
        st.write("All models were trained and evaluated on the exact same train/test split to guarantee fair comparison:")
        if not comp_df.empty:
            st.dataframe(
                comp_df.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color="#dbeafe"),
                use_container_width=True,
            )
        else:
            st.info("Comparison data not found. Run `python src/train.py` to generate.")

        # Comparison Plot
        comp_img = RESULTS_DIR / "model_comparison.png"
        if comp_img.exists():
            st.image(str(comp_img), caption="Comparative Model Performance Across Standard Metrics", use_container_width=True)

        st.divider()

        # Confusion Matrix & ROC Curve
        st.subheader("🔬 Confusion Matrix & ROC Curve (Test Set)")
        c1, c2 = st.columns(2)
        with c1:
            cm_img = RESULTS_DIR / "confusion_matrix.png"
            if cm_img.exists():
                st.image(str(cm_img), caption="Confusion Matrix on 205 Unseen Test Samples", use_container_width=True)
        with c2:
            roc_img = RESULTS_DIR / "roc_curve.png"
            if roc_img.exists():
                st.image(str(roc_img), caption="ROC Curve (Area Under Curve = 0.9928)", use_container_width=True)

        # Training History
        st.subheader("📉 Training & Validation Learning Curves")
        hist_img = RESULTS_DIR / "training_history.png"
        if hist_img.exists():
            st.image(str(hist_img), caption="Loss & Accuracy Curves across Training Epochs (Early Stopping at Best Epoch)", use_container_width=True)

    # =========================================================================
    # TAB 3: DATASET INSIGHTS & EDA
    # =========================================================================
    with tab_data:
        st.subheader("📊 Exploratory Data Analysis & Population Analytics")
        st.caption("Data source: `dataset/heart.csv` (1,025 observations across 13 clinical variables).")

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Patients", f"{len(df):,}")
        d2.metric("Clinical Features", f"{len(df.columns) - 1}")
        pos_cnt = int(df["target"].sum())
        d3.metric("Disease Cases (1)", f"{pos_cnt} ({pos_cnt / len(df) * 100:.1f}%)")
        neg_cnt = len(df) - pos_cnt
        d4.metric("Non-Disease Cases (0)", f"{neg_cnt} ({neg_cnt / len(df) * 100:.1f}%)")

        st.divider()

        eda_col1, eda_col2 = st.columns(2)
        with eda_col1:
            st.markdown("##### Age Distribution by Target")
            fig_age, ax_age = plt.subplots(figsize=(7, 4))
            sns.histplot(data=df, x="age", hue="target", kde=True, palette=["#3b82f6", "#ef4444"], bins=20, ax=ax_age)
            ax_age.set_title("Patient Age Distribution", fontsize=11, fontweight="bold")
            ax_age.set_xlabel("Age (years)")
            st.pyplot(fig_age)

        with eda_col2:
            st.markdown("##### Maximum Heart Rate (thalach) vs Disease")
            fig_hr, ax_hr = plt.subplots(figsize=(7, 4))
            sns.boxplot(data=df, x="target", y="thalach", palette=["#3b82f6", "#ef4444"], ax=ax_hr)
            ax_hr.set_xticklabels(["No Disease (0)", "Disease Present (1)"])
            ax_hr.set_title("Peak Heart Rate vs Heart Disease", fontsize=11, fontweight="bold")
            ax_hr.set_ylabel("Max Heart Rate (bpm)")
            st.pyplot(fig_hr)

        eda_col3, eda_col4 = st.columns(2)
        with eda_col3:
            st.markdown("##### Chest Pain Type Distribution")
            fig_cp, ax_cp = plt.subplots(figsize=(7, 4))
            sns.countplot(data=df, x="cp", hue="target", palette=["#3b82f6", "#ef4444"], ax=ax_cp)
            ax_cp.set_xticklabels(["Typical (0)", "Atypical (1)", "Non-anginal (2)", "Asymptomatic (3)"])
            ax_cp.set_title("Disease Incidence by Chest Pain Type", fontsize=11, fontweight="bold")
            ax_cp.set_xlabel("Chest Pain Type")
            st.pyplot(fig_cp)

        with eda_col4:
            st.markdown("##### Cholesterol vs Resting Blood Pressure")
            fig_scat, ax_scat = plt.subplots(figsize=(7, 4))
            sns.scatterplot(data=df, x="trestbps", y="chol", hue="target", palette=["#3b82f6", "#ef4444"], alpha=0.6, ax=ax_scat)
            ax_scat.set_title("Cholesterol vs Blood Pressure", fontsize=11, fontweight="bold")
            ax_scat.set_xlabel("Resting BP (mm Hg)")
            ax_scat.set_ylabel("Cholesterol (mg/dl)")
            st.pyplot(fig_scat)

        st.markdown("##### Dataset Inspection Preview")
        st.dataframe(df.head(10), use_container_width=True)

    # =========================================================================
    # TAB 4: MODEL EXPLAINABILITY
    # =========================================================================
    with tab_xai:
        st.subheader("🧠 Model Explainability & Clinical Risk Attribution")
        st.write(
            """
            Artificial Neural Networks are often described as 'black-box' models.
            To understand how our Multi-Layer Perceptron arrives at risk estimates,
            we compute **Permutation Feature Importance** directly on the held-out test set.
            """
        )

        feat_img = RESULTS_DIR / "feature_analysis.png"
        if feat_img.exists():
            st.image(str(feat_img), caption="Permutation Feature Importance (ROC-AUC Impact) & Correlation", use_container_width=True)

        st.info(
            """
            **How Permutation Importance Works:**
            Each clinical feature in the test set is randomly shuffled (breaking its relationship with the outcome)
            before passing through the preprocessing pipeline into the neural network. The resulting drop in ROC-AUC
            quantifies how strongly the model relies upon that specific clinical attribute.
            
            **Key Predictive Factors Identified:**
            - **Chest Pain Type (`cp`)**: Non-anginal and atypical presentations strongly drive probability increases.
            - **Major Vessels Colored (`ca`)**: Fluoroscopy vessel occlusion is a primary indicator of arterial stenosis.
            - **Thalassemia (`thal`) & Maximum Heart Rate (`thalach`)**: Significant markers for coronary perfusion adequacy.
            
            *Important Note: Statistical influence and feature importance reflect learned data patterns and do NOT constitute biological or medical causation.*
            """
        )

    # =========================================================================
    # TAB 5: PROJECT METHODOLOGY & ABOUT
    # =========================================================================
    with tab_about:
        st.subheader("ℹ️ Project Methodology & Academic Overview")
        st.markdown(
            """
            ### B.Tech Computer Science & Engineering (AI/ML) Project
            **Project Title**: Heart Disease Prediction using Artificial Neural Network (ANN)
            
            #### 1. End-to-End Pipeline Architecture
            1. **Data Ingestion**: Raw dataset loaded from `dataset/heart.csv`.
            2. **Data-Leakage Prevention**: Stratified 80/20 train/test split. Preprocessor is fitted strictly on training data.
            3. **Feature Preprocessing**:
               - `StandardScaler` applied to continuous variables (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`).
               - `OneHotEncoder` applied to multi-class categoricals (`cp`, `restecg`, `slope`, `ca`, `thal`).
               - Binary features passed through (`sex`, `fbs`, `exang`).
               - Pipeline serialized to `models/preprocessing_pipeline.pkl`.
            4. **Neural Network Architecture**:
               - Input Layer: 27 transformed features.
               - Dense Layer (64 neurons, ReLU, L2 Regularization = 0.001) + Batch Normalization + Dropout (0.3).
               - Dense Layer (32 neurons, ReLU, L2 Regularization = 0.001) + Batch Normalization + Dropout (0.2).
               - Dense Layer (16 neurons, ReLU) + Dropout (0.1).
               - Output Layer (1 neuron, Sigmoid activation for probability output).
            5. **Benchmarking**: Rigorous comparison with Logistic Regression, Decision Tree, Random Forest, and Baseline ANN.
            6. **Inference Interface**: Professional Streamlit web dashboard.

            #### 2. Limitations
            - Tabular dataset originates from standard Cleveland clinic cohort derivations (1,025 observations).
            - Real-world clinical adoption requires validation across multi-center patient populations and prospective cohorts.
            """
        )


if __name__ == "__main__":
    main()
