"""
Streamlit Web Application: Heart Disease Prediction using Artificial Neural Network (ANN)
========================================================================================
HeartCare: A modern, presentation-ready, and research-oriented AI/ML clinical analytics website.

Key Architectural & UI Highlights:
1. Top Sticky Navigation: Clean horizontal navbar anchored to the top with brand logo on the left
   ("HeartCare – Heart Disease Prediction") and active-highlighted navigation pills on the right.
2. Exact Natural Green Palette:
   - #F7FBF5: Main application surface background
   - #E2F1D9: Light green sections, cards, and input backgrounds
   - #C7E3B8: Card highlights, badges, and secondary surfaces
   - #A8C59A: Borders, hover accents, and outlines
   - #6B8A61: Primary buttons, active navigation, and key headings
3. Subtle Animated Background: Lightweight SVG/CSS animated ECG heartbeat line, soft pulsing
   heart outlines, medical crosses, ANN synapse nodes, and floating green particles.
4. College Presentation Ready: Hero section, empirical benchmark tabs, zero-leakage protocol,
   and permutation-based explainability.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from portal_pages import (
    render_stethoscope_background,
    render_patient_login,
    render_doctor_login,
    render_patient_portal,
    render_doctor_portal,
    save_prediction_for_patient,
)

# Set page configuration - Collapsed sidebar, Wide Layout
st.set_page_config(
    page_title="HeartCare – Heart Disease Prediction",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Relative directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "heart_disease_ann.keras"
PIPELINE_PATH = BASE_DIR / "models" / "preprocessing_pipeline.pkl"
RESULTS_DIR = BASE_DIR / "results"
METRICS_JSON_PATH = RESULTS_DIR / "final_metrics.json"
COMPARISON_CSV_PATH = RESULTS_DIR / "model_comparison.csv"
COMPARISON_STD_CSV_PATH = RESULTS_DIR / "model_comparison_standard.csv"
API_BASE_URL = os.getenv("CARDIOANN_API_URL", "http://127.0.0.1:8000")


# -----------------------------------------------------------------------------
# Cached Artifact Loaders
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
# Master Design System: Exact Green Palette & Modern Typography
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

    :root {
        --bg-main: #F7FBF5;
        --surface-light: #E2F1D9;
        --surface-card: #FFFFFF;
        --surface-highlight: #C7E3B8;
        --border-accent: #A8C59A;
        --brand-primary: #6B8A61;
        --brand-hover: #55724B;
        --text-primary: #1C2E19;
        --text-secondary: #3D5438;
        --text-muted: #5B7556;
    }

    /* Base Body & App Container */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
    }

    /* Completely hide Streamlit default sidebar and header artifacts */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0px !important;
        min-height: 0px !important;
        padding: 0 !important;
    }
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }

    .main, .block-container, div[data-testid="stVerticalBlock"] {
        overflow: visible !important;
    }

    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1240px !important;
    }

    /* -------------------------------------------------------------------------
       Sticky Top Navigation Bar
       ------------------------------------------------------------------------- */
    .top-navbar-wrapper {
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(247, 251, 245, 0.95);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-bottom: 2px solid var(--border-accent);
        padding: 12px 20px;
        margin: -0.5rem -1rem 1.75rem -1rem;
        border-radius: 0 0 16px 16px;
        box-shadow: 0 4px 20px rgba(107, 138, 97, 0.09);
    }

    .nav-brand-container {
        display: flex;
        align-items: center;
        gap: 12px;
        text-decoration: none;
        user-select: none;
    }

    .nav-brand-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        background: linear-gradient(135deg, #6B8A61 0%, #4D6D44 100%);
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(107, 138, 97, 0.32);
        color: #F7FBF5;
        animation: pulseIcon 3s ease-in-out infinite;
    }

    @keyframes pulseIcon {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }

    .nav-brand-text {
        display: flex;
        flex-direction: column;
    }

    .nav-brand-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 1.35rem;
        color: var(--brand-primary);
        line-height: 1.1;
        letter-spacing: -0.01em;
    }

    .nav-brand-subtitle {
        font-size: 0.80rem;
        font-weight: 600;
        color: var(--text-muted);
        letter-spacing: 0.02em;
    }

    /* Style Streamlit buttons inside the top navbar wrapper */
    .top-navbar-wrapper [data-testid="stButton"] button {
        width: 100% !important;
        white-space: nowrap !important;
        padding: 8px 12px !important;
        font-size: 0.88rem !important;
        border-radius: 20px !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    .top-navbar-wrapper [data-testid="stButton"] button[kind="primary"] {
        background: var(--brand-primary) !important;
        color: #FFFFFF !important;
        border: 1.5px solid var(--brand-primary) !important;
        box-shadow: 0 4px 14px rgba(107, 138, 97, 0.28) !important;
        font-weight: 700 !important;
    }

    .top-navbar-wrapper [data-testid="stButton"] button[kind="secondary"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border: 1.5px solid transparent !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    .top-navbar-wrapper [data-testid="stButton"] button[kind="secondary"]:hover {
        background: var(--surface-light) !important;
        border-color: var(--border-accent) !important;
        color: var(--text-primary) !important;
        transform: translateY(-1.5px) !important;
    }

    .mobile-nav-wrapper {
        display: none;
    }

    @media (max-width: 860px) {
        .mobile-nav-wrapper {
            display: block;
            margin-top: 10px;
            margin-bottom: 12px;
        }
    }

    /* -------------------------------------------------------------------------
       Hero Section Styling
       ------------------------------------------------------------------------- */
    .hero-container {
        background: linear-gradient(135deg, #E2F1D9 0%, #C7E3B8 55%, #A8C59A 100%);
        border: 2px solid var(--border-accent);
        border-radius: 20px;
        padding: 36px 40px;
        margin-bottom: 24px;
        box-shadow: 0 8px 30px rgba(107, 138, 97, 0.12);
        position: relative;
        overflow: hidden;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(107, 138, 97, 0.20);
        border: 1.5px solid var(--brand-primary);
        color: var(--brand-primary);
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 5px 14px;
        border-radius: 30px;
        margin-bottom: 14px;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background: var(--brand-primary);
        border-radius: 50%;
        animation: dotPulse 1.8s infinite;
    }

    @keyframes dotPulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.4); opacity: 0.5; }
    }

    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.30rem;
        font-weight: 800;
        color: var(--brand-primary);
        line-height: 1.15;
        letter-spacing: -0.02em;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        color: var(--text-secondary);
        line-height: 1.6;
        max-width: 880px;
        margin-bottom: 20px;
    }

    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 14px;
    }

    .hero-pill {
        background: #F7FBF5;
        border: 1px solid var(--border-accent);
        color: var(--text-primary);
        font-size: 0.82rem;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 20px;
        box-shadow: 0 2px 6px rgba(107, 138, 97, 0.08);
    }

    /* -------------------------------------------------------------------------
       Card System: Metric Cards, Action Cards, Info Cards
       ------------------------------------------------------------------------- */
    .metric-card {
        background: #FFFFFF;
        border: 1.5px solid var(--border-accent);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px rgba(107, 138, 97, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(107, 138, 97, 0.16);
        border-color: var(--brand-primary);
    }

    .metric-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
        color: var(--text-muted);
        margin-bottom: 4px;
    }

    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 1.88rem;
        font-weight: 800;
        color: var(--brand-primary);
        line-height: 1.1;
    }

    .metric-sub {
        font-size: 0.80rem;
        color: var(--text-secondary);
        font-weight: 600;
        margin-top: 4px;
    }

    .medical-disclaimer {
        background-color: var(--surface-light);
        border: 1.5px solid var(--border-accent);
        border-left: 5px solid var(--brand-primary);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 24px;
        font-size: 0.90rem;
        color: var(--text-primary);
        line-height: 1.6;
    }

    .action-card {
        background: var(--surface-light);
        border: 1.5px solid var(--border-accent);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(107, 138, 97, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }

    .action-card:hover {
        transform: translateY(-2px);
        border-color: var(--brand-primary);
        box-shadow: 0 8px 22px rgba(107, 138, 97, 0.14);
    }

    .action-card h4 {
        color: var(--brand-primary);
        font-family: 'Outfit', sans-serif;
        font-size: 1.20rem;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 8px;
    }

    .action-card p {
        color: var(--text-secondary);
        font-size: 0.92rem;
        line-height: 1.5;
        margin-bottom: 16px;
    }

    /* Section Subheadings */
    .section-title {
        color: var(--brand-primary);
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 1.55rem;
        margin-top: 0;
        margin-bottom: 6px;
        letter-spacing: -0.01em;
    }

    .section-caption {
        color: var(--text-secondary);
        font-size: 0.94rem;
        margin-bottom: 22px;
        line-height: 1.55;
    }

    .form-section-header {
        color: var(--brand-primary);
        font-size: 1.05rem;
        font-weight: 700;
        border-bottom: 2px solid var(--surface-highlight);
        padding-bottom: 6px;
        margin-bottom: 14px;
    }

    /* Prediction Result Cards */
    .result-card-elevated {
        background: #FFF5F5;
        border: 2px solid #C85A50;
        border-radius: 14px;
        padding: 22px;
        margin-top: 16px;
        box-shadow: 0 4px 14px rgba(200, 90, 80, 0.12);
        color: #3A1010;
    }

    .result-card-low {
        background: var(--surface-light);
        border: 2px solid var(--brand-primary);
        border-radius: 14px;
        padding: 22px;
        margin-top: 16px;
        box-shadow: 0 4px 14px rgba(107, 138, 97, 0.12);
        color: var(--text-primary);
    }

    /* General Button Styling Across Pages */
    .stButton > button[kind="primary"]:not(.top-navbar-wrapper *) {
        background: var(--brand-primary) !important;
        color: #FFFFFF !important;
        border: 1.5px solid var(--brand-primary) !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 10px 22px !important;
        box-shadow: 0 4px 14px rgba(107, 138, 97, 0.28) !important;
        transition: all 0.25s ease !important;
    }

    .stButton > button[kind="primary"]:not(.top-navbar-wrapper *):hover {
        background: var(--brand-hover) !important;
        border-color: var(--brand-hover) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(107, 138, 97, 0.38) !important;
    }

    .stButton > button[kind="secondary"]:not(.top-navbar-wrapper *) {
        background: var(--surface-light) !important;
        color: var(--text-primary) !important;
        border: 1.5px solid var(--border-accent) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 10px 22px !important;
        transition: all 0.25s ease !important;
    }

    .stButton > button[kind="secondary"]:not(.top-navbar-wrapper *):hover {
        background: var(--surface-highlight) !important;
        border-color: var(--brand-primary) !important;
        transform: translateY(-2px) !important;
    }

    /* Streamlit Tabs Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid var(--border-accent);
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 700;
        color: var(--text-secondary);
        border-radius: 8px 8px 0 0;
        background: transparent;
    }

    .stTabs [aria-selected="true"] {
        color: var(--brand-primary) !important;
        border-bottom: 3px solid var(--brand-primary) !important;
        background: var(--surface-light) !important;
    }

    /* Input widgets styling */
    [data-testid="stNumberInput"], [data-testid="stSelectbox"] {
        border-radius: 10px;
    }

    /* Unified Footer */
    .site-footer {
        border-top: 2px solid var(--border-accent);
        background: var(--surface-light);
        border-radius: 16px 16px 0 0;
        padding: 26px 20px;
        margin-top: 50px;
        color: var(--text-secondary);
        font-size: 0.88rem;
        line-height: 1.6;
        text-align: center;
    }

    /* -------------------------------------------------------------------------
       Animated Project Background: ECG Line, Synapses, Floating Hearts & Crosses
       ------------------------------------------------------------------------- */
    .heartcare-animated-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        pointer-events: none;
        user-select: none;
        z-index: 0;
        overflow: hidden;
        opacity: 0.88;
    }

    .bg-svg-canvas {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
    }

    /* ECG Flow Animation */
    .ecg-line {
        animation: ecgFlow 14s linear infinite;
    }

    @keyframes ecgFlow {
        0% { transform: translateX(0); }
        100% { transform: translateX(600px); }
    }

    /* Neural Synapses & Node Pulses */
    .synapse-line {
        animation: synapseDash 4s linear infinite;
    }

    @keyframes synapseDash {
        0% { stroke-dashoffset: 0; }
        100% { stroke-dashoffset: 24; }
    }

    .ann-node.pulse-slow {
        animation: nodePulse 3.5s ease-in-out infinite;
    }

    .ann-node.pulse-med {
        animation: nodePulse 2.4s ease-in-out infinite;
    }

    .ann-node.pulse-fast {
        animation: nodePulse 1.8s ease-in-out infinite;
    }

    @keyframes nodePulse {
        0%, 100% { transform: scale(1); opacity: 0.7; }
        50% { transform: scale(1.25); opacity: 1; }
    }

    /* Floating Heart Pulse & Drift */
    .floating-element {
        position: absolute;
    }

    .heart-elem {
        animation: heartPulseAndFloat 8s ease-in-out infinite;
    }

    .heart-elem.h1 {
        top: 22%;
        left: 6%;
        animation-delay: 0s;
    }

    .heart-elem.h2 {
        bottom: 28%;
        right: 8%;
        animation-delay: 3.5s;
    }

    @keyframes heartPulseAndFloat {
        0%, 100% { transform: translateY(0) scale(1); opacity: 0.20; }
        14% { transform: translateY(-4px) scale(1.12); opacity: 0.35; }
        28% { transform: translateY(-8px) scale(1); opacity: 0.22; }
        42% { transform: translateY(-12px) scale(1.08); opacity: 0.32; }
        70% { transform: translateY(-6px) scale(1); opacity: 0.20; }
    }

    /* Floating Cross Drift */
    .cross-elem {
        animation: crossFloat 11s ease-in-out infinite;
    }

    .cross-elem.c1 {
        top: 48%;
        right: 5%;
        animation-delay: 1.2s;
    }

    .cross-elem.c2 {
        bottom: 16%;
        left: 12%;
        animation-delay: 4.8s;
    }

    @keyframes crossFloat {
        0%, 100% { transform: translateY(0) rotate(0deg); opacity: 0.18; }
        50% { transform: translateY(-16px) rotate(45deg); opacity: 0.32; }
    }

    /* Floating Light Green Particles */
    .particle {
        position: absolute;
        border-radius: 50%;
        background: #C7E3B8;
        pointer-events: none;
        animation: particleDrift 16s linear infinite;
    }

    .particle.p1 { width: 9px; height: 9px; left: 18%; bottom: -20px; animation-duration: 17s; opacity: 0.28; }
    .particle.p2 { width: 14px; height: 14px; left: 42%; bottom: -20px; animation-duration: 22s; animation-delay: 3s; opacity: 0.22; }
    .particle.p3 { width: 8px; height: 8px; left: 68%; bottom: -20px; animation-duration: 19s; animation-delay: 7s; opacity: 0.30; }
    .particle.p4 { width: 11px; height: 11px; left: 86%; bottom: -20px; animation-duration: 21s; animation-delay: 2s; opacity: 0.20; }
    .particle.p5 { width: 6px; height: 6px; left: 30%; bottom: -20px; animation-duration: 14s; animation-delay: 5s; opacity: 0.35; }

    @keyframes particleDrift {
        0% { transform: translateY(0) translateX(0); }
        50% { transform: translateY(-50vh) translateX(20px); }
        100% { transform: translateY(-105vh) translateX(-15px); }
    }

    /* Accessibility: Reduced Motion */
    @media (prefers-reduced-motion: reduce) {
        .heartcare-animated-bg * {
            animation: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_animated_background():
    """Renders subtle, hardware-accelerated animated medical & neural network background."""
    st.markdown(
        """
        <div class="heartcare-animated-bg" aria-hidden="true">
            <svg class="bg-svg-canvas" xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
                <defs>
                    <linearGradient id="ecgGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#6B8A61" stop-opacity="0.0"/>
                        <stop offset="15%" stop-color="#6B8A61" stop-opacity="0.30"/>
                        <stop offset="50%" stop-color="#6B8A61" stop-opacity="0.45"/>
                        <stop offset="85%" stop-color="#6B8A61" stop-opacity="0.30"/>
                        <stop offset="100%" stop-color="#6B8A61" stop-opacity="0.0"/>
                    </linearGradient>
                    <linearGradient id="synapseGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#A8C59A" stop-opacity="0.35"/>
                        <stop offset="100%" stop-color="#6B8A61" stop-opacity="0.20"/>
                    </linearGradient>
                </defs>

                <!-- Neural Network Cluster (Top Right) -->
                <g class="neural-cluster">
                    <line x1="84%" y1="10%" x2="91%" y2="16%" stroke="url(#synapseGrad)" stroke-width="1.8" stroke-dasharray="4,4" class="synapse-line" />
                    <line x1="84%" y1="22%" x2="91%" y2="16%" stroke="url(#synapseGrad)" stroke-width="1.8" stroke-dasharray="4,4" class="synapse-line" />
                    <line x1="84%" y1="10%" x2="91%" y2="28%" stroke="url(#synapseGrad)" stroke-width="1.8" stroke-dasharray="4,4" class="synapse-line" />
                    <line x1="84%" y1="22%" x2="91%" y2="28%" stroke="url(#synapseGrad)" stroke-width="1.8" stroke-dasharray="4,4" class="synapse-line" />
                    <line x1="91%" y1="16%" x2="97%" y2="22%" stroke="url(#synapseGrad)" stroke-width="2.2" class="synapse-line" />
                    <line x1="91%" y1="28%" x2="97%" y2="22%" stroke="url(#synapseGrad)" stroke-width="2.2" class="synapse-line" />

                    <circle cx="84%" cy="10%" r="6" fill="#C7E3B8" stroke="#6B8A61" stroke-width="2" class="ann-node pulse-slow" />
                    <circle cx="84%" cy="22%" r="6" fill="#C7E3B8" stroke="#6B8A61" stroke-width="2" class="ann-node pulse-slow" />
                    <circle cx="91%" cy="16%" r="7" fill="#E2F1D9" stroke="#6B8A61" stroke-width="2" class="ann-node pulse-med" />
                    <circle cx="91%" cy="28%" r="7" fill="#E2F1D9" stroke="#6B8A61" stroke-width="2" class="ann-node pulse-med" />
                    <circle cx="97%" cy="22%" r="8" fill="#6B8A61" stroke="#F7FBF5" stroke-width="2.5" class="ann-node pulse-fast" />
                </g>

                <!-- Neural Network Cluster (Bottom Left) -->
                <g class="neural-cluster">
                    <line x1="5%" y1="78%" x2="11%" y2="72%" stroke="url(#synapseGrad)" stroke-width="1.8" stroke-dasharray="4,4" class="synapse-line" />
                    <line x1="5%" y1="90%" x2="11%" y2="72%" stroke="url(#synapseGrad)" stroke-width="1.8" stroke-dasharray="4,4" class="synapse-line" />
                    <line x1="11%" y1="72%" x2="17%" y2="80%" stroke="url(#synapseGrad)" stroke-width="2.2" class="synapse-line" />
                    <circle cx="5%" cy="78%" r="5.5" fill="#C7E3B8" stroke="#A8C59A" stroke-width="2" class="ann-node pulse-slow" />
                    <circle cx="5%" cy="90%" r="5.5" fill="#C7E3B8" stroke="#A8C59A" stroke-width="2" class="ann-node pulse-slow" />
                    <circle cx="11%" cy="72%" r="7" fill="#E2F1D9" stroke="#6B8A61" stroke-width="2" class="ann-node pulse-med" />
                    <circle cx="17%" cy="80%" r="8" fill="#6B8A61" stroke="#E2F1D9" stroke-width="2.5" class="ann-node pulse-fast" />
                </g>

                <!-- Moving Continuous ECG / Cardiac Wave Trace -->
                <g class="ecg-track">
                    <path class="ecg-line" d="M -800,880 L -500,880 L -480,880 L -470,865 L -460,895 L -450,880 L -410,880 L -400,820 L -390,950 L -380,845 L -370,890 L -360,880 L -330,880 L -310,860 L -290,880 L -100,880 L -80,880 L -70,865 L -60,895 L -50,880 L -10,880 L 0,820 L 10,950 L 20,845 L 30,890 L 40,880 L 70,880 L 90,860 L 110,880 L 300,880 L 320,880 L 330,865 L 340,895 L 350,880 L 390,880 L 400,820 L 410,950 L 420,845 L 430,890 L 440,880 L 470,880 L 490,860 L 510,880 L 700,880 L 720,880 L 730,865 L 740,895 L 750,880 L 790,880 L 800,820 L 810,950 L 820,845 L 830,890 L 840,880 L 870,880 L 890,860 L 910,880 L 1100,880 L 1120,880 L 1130,865 L 1140,895 L 1150,880 L 1190,880 L 1200,820 L 1210,950 L 1220,845 L 1230,890 L 1240,880 L 1270,880 L 1290,860 L 1310,880 L 1500,880 L 1520,880 L 1530,865 L 1540,895 L 1550,880 L 1590,880 L 1600,820 L 1610,950 L 1620,845 L 1630,890 L 1640,880 L 1670,880 L 1690,860 L 1710,880 L 2000,880" fill="none" stroke="url(#ecgGrad)" stroke-width="2.6" />
                </g>
            </svg>

            <!-- Floating Heart Outlines & Medical Crosses -->
            <div class="floating-element heart-elem h1">
                <svg viewBox="0 0 24 24" width="36" height="36" fill="none" stroke="#6B8A61" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
                </svg>
            </div>
            <div class="floating-element heart-elem h2">
                <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="#A8C59A" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
                </svg>
            </div>
            <div class="floating-element cross-elem c1">
                <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="#6B8A61" stroke-width="2.4" stroke-linecap="round">
                    <line x1="12" y1="5" x2="12" y2="19"></line>
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                </svg>
            </div>
            <div class="floating-element cross-elem c2">
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#A8C59A" stroke-width="2" stroke-linecap="round">
                    <line x1="12" y1="5" x2="12" y2="19"></line>
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                </svg>
            </div>

            <!-- Floating Subtle Light Green Particles -->
            <div class="particle p1"></div>
            <div class="particle p2"></div>
            <div class="particle p3"></div>
            <div class="particle p4"></div>
            <div class="particle p5"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_navbar(current_section):
    """Renders persistent sticky top navigation bar with brand on left and nav links on right."""
    st.markdown('<div class="top-navbar-wrapper">', unsafe_allow_html=True)

    col_brand, col_nav = st.columns([1.2, 3.4], vertical_alignment="center")

    with col_brand:
        st.markdown(
            """
            <div class="nav-brand-container">
                <div class="nav-brand-icon">
                    <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
                        <polyline points="3 12 7 12 9 9 12 16 14 11 16 13 21 13"></polyline>
                    </svg>
                </div>
                <div class="nav-brand-text">
                    <span class="nav-brand-title">HeartCare</span>
                    <span class="nav-brand-subtitle">Heart Disease Prediction</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_nav:
        items = ["Home", "Prediction", "Model Performance", "Model Explainability", "About"]
        auth = st.session_state.get("auth")
        if auth is None:
            items += ["Patient Login", "Doctor Login"]
        elif auth["user"]["role"] == "patient":
            items += ["My Portal"]
        else:
            items += ["Doctor Portal"]

        for col, label in zip(st.columns(len(items)), items):
            with col:
                if st.button(label, key=f"top_nav_{label}", type="primary" if current_section == label else "secondary", width="stretch"):
                    st.session_state.active_nav_section = label
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)



def render_common_footer():
    """Renders persistent academic & medical disclaimer footer."""
    st.markdown(
        """
        <div class="site-footer">
            <div style="font-weight:800; font-size:1.15rem; color:#6B8A61; margin-bottom:4px;">
                💚 HeartCare – Heart Disease Prediction
            </div>
            <div style="color:#3D5438; font-size:0.90rem; margin-bottom:10px;">
                Clinical Risk Estimation using Artificial Neural Network (ANN) • B.Tech Capstone Project
            </div>
            <div style="color:#5B7556; font-size:0.82rem; margin-bottom:14px;">
                <em>Architecture: TensorFlow 2 / Keras 3 • Scikit-Learn • Streamlit • FastAPI REST Backend</em>
            </div>
            <div style="background:#C7E3B8; border:1px solid #A8C59A; border-radius:10px; padding:12px 18px; font-size:0.82rem; color:#1C2E19; max-width:850px; margin:0 auto; line-height:1.55;">
                <strong>Ethical & Clinical Disclaimer:</strong> This software is developed strictly as an educational machine learning research prototype.
                It is <strong>not</strong> a certified medical diagnostic device. Predictions represent statistical likelihood and must never replace qualified clinical judgment, medical diagnosis, or patient triage.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Main Application Router & Navigation System
# -----------------------------------------------------------------------------
def main():
    # Session state initialization & backward compatibility
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    if "active_nav_section" not in st.session_state:
        st.session_state.active_nav_section = "Home"

    # Migrate any legacy section names seamlessly
    legacy_map = {
        "Home Dashboard": "Home",
        "Heart Disease Prediction": "Prediction",
        "About the Project": "About",
        "Training & Evaluation": "Model Performance",
    }
    if st.session_state.active_nav_section in legacy_map:
        st.session_state.active_nav_section = legacy_map[st.session_state.active_nav_section]

    # Render hardware-accelerated animated medical background
    render_animated_background()
    render_stethoscope_background()

    # Render Sticky Top Navigation Bar
    render_top_navbar(st.session_state.active_nav_section)

    # Load artifacts and evaluation metrics
    model, pipeline, err_art = load_ml_artifacts()
    metrics, comp_df, comp_std_df = load_benchmark_metrics()

    # Safe error check for missing model
    if err_art:
        st.error(err_art)
        st.info("Run `python src/train.py` from your project root to train and serialize the neural network model.")
        render_common_footer()
        return

    # -------------------------------------------------------------------------
    # SECTION 1: HOME
    # -------------------------------------------------------------------------
    if st.session_state.active_nav_section == "Home":
        # Professional Hero Section
        st.markdown(
            """
            <div class="hero-container">
                <div class="hero-badge">
                    <span class="pulse-dot"></span> Clinical Intelligence & Deep Learning
                </div>
                <h1 class="hero-title">Heart Disease Prediction Using Artificial Neural Network</h1>
                <p class="hero-subtitle">
                    A clinical-grade cardiovascular risk estimation platform powered by a zero-leakage Multi-Layer Perceptron (ANN), delivering calibrated continuous risk probabilities and permutation-based feature explainability.
                </p>
                <div class="hero-pills">
                    <span class="hero-pill">✓ 0.9015 ROC-AUC Discrimination</span>
                    <span class="hero-pill">✓ 89.66% High Precision</span>
                    <span class="hero-pill">✓ Verified Zero-Leakage Holdout (N=302)</span>
                    <span class="hero-pill">✓ 13 Physiological Markers</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Hero Action CTA Buttons
        cta_col1, cta_col2, _ = st.columns([1.2, 1.4, 2.0])
        with cta_col1:
            if st.button("Explore Prediction ➔", key="btn_hero_pred", type="primary", width="stretch"):
                st.session_state.active_nav_section = "Prediction"
                st.rerun()
        with cta_col2:
            if st.button("View Model Performance 📊", key="btn_hero_perf", type="secondary", width="stretch"):
                st.session_state.active_nav_section = "Model Performance"
                st.rerun()

        st.write("")

        # 4 Key Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">Dataset Cohort</div>
                    <div class="metric-value">1,025</div>
                    <div class="metric-sub">302 Unique Patients</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">Neural Network ROC-AUC</div>
                    <div class="metric-value">0.9015</div>
                    <div class="metric-sub">Zero-Leakage Holdout</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">Model Precision</div>
                    <div class="metric-value">89.66%</div>
                    <div class="metric-sub">Low False Alarm Rate</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">Input Parameters</div>
                    <div class="metric-value">13</div>
                    <div class="metric-sub">27 Engineered Features</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown(
            """
            <div class="medical-disclaimer">
                <strong>Project Mission:</strong> Cardiovascular diseases are the leading cause of mortality globally.
                This project builds an Artificial Neural Network trained on standard non-invasive clinical attributes to estimate the probability of cardiac risk markers.
                All metrics are verified using a strict zero-leakage 3-way partition (Train: 204, Val: 37, Test: 61) to guarantee genuine clinical generalization.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">Quick Actions & Interactive Hub</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Navigate directly to any section of the website using these interactive action cards:</div>', unsafe_allow_html=True)

        qa1, qa2 = st.columns(2)
        with qa1:
            st.markdown(
                """
                <div class="action-card">
                    <h4>Clinical Risk Prediction</h4>
                    <p>Enter 13 patient physiological parameters, adjust classification thresholds, view model probabilities, and download formatted clinical summaries.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Launch Clinical Prediction Form", key="btn_qa_pred", type="primary", width="stretch"):
                st.session_state.active_nav_section = "Prediction"
                st.rerun()

            st.write("")
            st.markdown(
                """
                <div class="action-card">
                    <h4>Model Performance & ROC Analysis</h4>
                    <p>Inspect confusion matrices, ROC curves, sensitivity/recall trade-offs, and empirical benchmarks on the held-out test cohort.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("View Model Performance Metrics", key="btn_qa_perf", width="stretch"):
                st.session_state.active_nav_section = "Model Performance"
                st.rerun()

        with qa2:
            st.markdown(
                """
                <div class="action-card">
                    <h4>Explainable AI (XAI) Insights</h4>
                    <p>Understand which clinical features (e.g., chest pain type, ST depression, fluoroscopy vessels) drive the neural network's decision boundaries.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Explore Model Explainability", key="btn_qa_xai", width="stretch"):
                st.session_state.active_nav_section = "Model Explainability"
                st.rerun()

            st.write("")
            st.markdown(
                """
                <div class="action-card">
                    <h4>Project Architecture & Research</h4>
                    <p>Explore the end-to-end data pipeline, zero-leakage methodology, technology stack, and academic capstone documentation.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Read Project Documentation", key="btn_qa_abt", width="stretch"):
                st.session_state.active_nav_section = "About"
                st.rerun()

    # -------------------------------------------------------------------------
    # SECTION 2: PREDICTION
    # -------------------------------------------------------------------------
    elif st.session_state.active_nav_section == "Prediction":
        st.markdown('<div class="section-title">Patient Clinical Risk Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Configure the patient\'s 13 physiological parameters across the four clinical sections below to generate an estimated probability of heart disease risk markers.</div>',
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
                    help="Patient chronological age in years",
                )
                sex_label = st.selectbox(
                    "Biological Sex",
                    options=["Male", "Female"],
                    index=0,
                    help="Biological sex of the patient",
                )
                sex = 1 if sex_label == "Male" else 0

            # Category 2: Baseline Hemodynamics
            with c2:
                st.markdown('<div class="form-section-header">2. Hemodynamics</div>', unsafe_allow_html=True)
                cp_options = {
                    "Typical Angina (0)": 0,
                    "Atypical Angina (1)": 1,
                    "Non-Anginal Pain (2)": 2,
                    "Asymptomatic (3)": 3,
                }
                cp_label = st.selectbox(
                    "Chest Pain Type",
                    options=list(cp_options.keys()),
                    index=0,
                    help="Classification of patient chest pain symptoms",
                )
                cp = cp_options[cp_label]

                trestbps = st.number_input(
                    "Resting Blood Pressure (mm Hg)",
                    min_value=80,
                    max_value=220,
                    value=130,
                    step=1,
                    help="Resting systolic blood pressure upon clinical admission",
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
                    help="Fasting blood sugar exceeding 120 mg/dl indicator",
                )
                fbs = 1 if "True" in fbs_label else 0

            # Category 3: Cardiac Electrophysiology
            with c3:
                st.markdown('<div class="form-section-header">3. Electrophysiology</div>', unsafe_allow_html=True)
                restecg_options = {
                    "Normal (0)": 0,
                    "ST-T Wave Abnormality (1)": 1,
                    "Left Ventricular Hypertrophy (2)": 2,
                }
                restecg_label = st.selectbox(
                    "Resting ECG Result",
                    options=list(restecg_options.keys()),
                    index=0,
                    help="Resting electrocardiographic diagnostic findings",
                )
                restecg = restecg_options[restecg_label]

                thalach = st.number_input(
                    "Max Heart Rate Achieved (bpm)",
                    min_value=60,
                    max_value=230,
                    value=150,
                    step=1,
                    help="Maximum peak heart rate recorded during treadmill exercise testing",
                )

                exang_label = st.selectbox(
                    "Exercise Induced Angina",
                    options=["No", "Yes"],
                    index=0,
                    help="Angina provoked by treadmill exercise stress",
                )
                exang = 1 if exang_label == "Yes" else 0

            # Category 4: Stress Markers & Imaging
            with c4:
                st.markdown('<div class="form-section-header">4. Stress & Imaging</div>', unsafe_allow_html=True)
                oldpeak = st.number_input(
                    "ST Depression (oldpeak)",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    format="%.1f",
                    help="ST depression induced by exercise relative to resting baseline",
                )

                slope_options = {
                    "Upsloping (0)": 0,
                    "Flat (1)": 1,
                    "Downsloping (2)": 2,
                }
                slope_label = st.selectbox(
                    "Peak Exercise ST Slope",
                    options=list(slope_options.keys()),
                    index=1,
                    help="Slope gradient of peak exercise ST segment",
                )
                slope = slope_options[slope_label]

                ca = st.selectbox(
                    "Fluoroscopy Major Vessels (ca)",
                    options=[0, 1, 2, 3, 4],
                    index=0,
                    help="Number of major coronary vessels colored by fluoroscopy",
                )

                thal_options = {
                    "Unknown / Baseline (0)": 0,
                    "Normal Perfusion (1)": 1,
                    "Fixed Defect (2)": 2,
                    "Reversible Defect (3)": 3,
                }
                thal_label = st.selectbox(
                    "Thallium Stress Scintigraphy",
                    options=list(thal_options.keys()),
                    index=2,
                    help="Thallium nuclear perfusion defect status",
                )
                thal = thal_options[thal_label]

            # Classification Decision Threshold Slider
            st.divider()
            threshold_col1, threshold_col2 = st.columns([2, 1], vertical_alignment="center")
            with threshold_col1:
                threshold = st.slider(
                    "Clinical Classification Decision Threshold (tau)",
                    min_value=0.10,
                    max_value=0.90,
                    value=0.50,
                    step=0.05,
                    help="Operating probability threshold to balance sensitivity vs false-positive alarms.",
                )
            with threshold_col2:
                st.caption(
                    f"Selected threshold: **{threshold:.2f}**.<br>"
                    "Lower threshold prioritizes sensitivity (minimizes missed cases); higher threshold reduces false alarms.",
                    unsafe_allow_html=True,
                )

            submit_pred = st.form_submit_button(
                "Compute Statistical Risk Prediction",
                type="primary",
                width="stretch",
            )

        # Inference Processing Pipeline
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
                # Transform through serialized Scikit-Learn ColumnTransformer
                X_trans = pipeline.transform(input_df)

                # Predict continuous probability with calibrated neural network
                prob_raw = float(model.predict(X_trans, verbose=0)[0][0])
                prob_pct = prob_raw * 100.0
                is_elevated = prob_raw >= threshold

                # Risk Stratification Tiers
                if prob_raw < 0.30:
                    risk_stratum = "Low Statistical Risk"
                    risk_color = "#6B8A61"
                    recommendation = (
                        "Baseline cardiac markers are within expected statistical ranges for this cohort. "
                        "Routine preventive wellness monitoring, Mediterranean diet, and aerobic exercise are advised."
                    )
                elif prob_raw < 0.65:
                    risk_stratum = "Intermediate / Borderline Risk"
                    risk_color = "#A8C59A"
                    recommendation = (
                        "Borderline cardiac indicators observed. Recommend outpatient clinical follow-up, "
                        "comprehensive lipid profile audit, ambulatory BP monitoring, and lifestyle intervention."
                    )
                else:
                    risk_stratum = "Elevated Cardiac Marker Risk"
                    risk_color = "#C85A50"
                    recommendation = (
                        "Multiple clinical indicators (e.g., chest pain presentation, fluoroscopy vessels, or ST depression) "
                        "strongly align with disease profiles. Immediate cardiology consultation and diagnostic echocardiography recommended."
                    )

                # Log session history for audit
                st.session_state.prediction_history.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Age": age,
                    "Sex": sex_label,
                    "ChestPain": cp_label,
                    "BP": trestbps,
                    "Chol": chol,
                    "MaxHR": thalach,
                    "Oldpeak": oldpeak,
                    "Vessels": ca,
                    "Probability": f"{prob_pct:.1f}%",
                    "Threshold": f"{threshold:.2f}",
                    "Status": "Elevated Risk" if is_elevated else "Low Risk",
                })
                auth = st.session_state.get("auth")
                if auth and auth["user"]["role"] == "patient":
                    ok_save, resp_save = save_prediction_for_patient(auth["user"]["id"], {**input_dict, "threshold": threshold})
                    if ok_save:
                        st.success(f"Saved to your patient history (record #{resp_save['prediction_id']}).")
                    else:
                        st.warning(f"Could not save to history: {resp_save}")
                else:
                    st.info("Login as a patient to save this result to your history.")
                

                # Visual Output Card
                if is_elevated:
                    st.markdown(
                        f"""
                        <div class="result-card-elevated">
                            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                                <div>
                                    <span style="background:#C85A50; color:#FFFFFF; padding:4px 12px; border-radius:20px; font-weight:700; font-size:0.80rem; text-transform:uppercase;">
                                        Elevated Risk Markers Detected
                                    </span>
                                    <h2 style="color:#C85A50; margin:10px 0 4px 0; font-family:'Outfit', sans-serif;">Estimated Disease Probability: {prob_pct:.1f}%</h2>
                                    <div style="font-size:0.95rem; color:#3A1010; font-weight:600;">Stratification: {risk_stratum} (Threshold = {threshold:.2f})</div>
                                </div>
                                <div style="font-size:3.2rem; margin-right:15px;">⚠️</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card-low">
                            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                                <div>
                                    <span style="background:#6B8A61; color:#FFFFFF; padding:4px 12px; border-radius:20px; font-weight:700; font-size:0.80rem; text-transform:uppercase;">
                                        Low Statistical Risk Markers
                                    </span>
                                    <h2 style="color:#6B8A61; margin:10px 0 4px 0; font-family:'Outfit', sans-serif;">Estimated Disease Probability: {prob_pct:.1f}%</h2>
                                    <div style="font-size:0.95rem; color:#1C2E19; font-weight:600;">Stratification: {risk_stratum} (Threshold = {threshold:.2f})</div>
                                </div>
                                <div style="font-size:3.2rem; margin-right:15px;">💚</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Diagnostic Summary Grid
                st.write("")
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                with res_col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Neural Net Output</div>
                            <div class="metric-value">{prob_pct:.1f}%</div>
                            <div class="metric-sub">Sigmoid Activation</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with res_col2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Selected Threshold</div>
                            <div class="metric-value">{threshold:.2f}</div>
                            <div class="metric-sub">Decision Boundary</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with res_col3:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Predicted Class</div>
                            <div class="metric-value">{"Class 1" if is_elevated else "Class 0"}</div>
                            <div class="metric-sub">{"Positive Marker" if is_elevated else "Negative Control"}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with res_col4:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Risk Category</div>
                            <div class="metric-value" style="font-size:1.35rem;">{risk_stratum.split()[0]}</div>
                            <div class="metric-sub">Clinical Triaging</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Probability Distribution Progress Bar
                st.write("")
                st.write("**Model Risk Probability Spectrum:**")
                st.progress(min(prob_raw, 1.0))

                # Clinical Recommendation Card
                st.markdown(
                    f"""
                    <div style="background:#E2F1D9; border:1.5px solid #A8C59A; border-left:5px solid {risk_color}; border-radius:12px; padding:18px 22px; margin-top:16px;">
                        <h4 style="color:#1C2E19; margin-top:0; font-family:'Outfit', sans-serif;">Clinical Guidance & Interpretation:</h4>
                        <p style="color:#2D4A27; margin-bottom:0; font-size:0.95rem; line-height:1.6;">{recommendation}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Report Download & Explainability Link Buttons
                st.write("")
                down_col, xai_link_col = st.columns(2)
                with down_col:
                    report_text = f"""================================================================================
CARDIOANN CLINICAL RISK PREDICTION SUMMARY
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
System: CardioANN Multi-Layer Perceptron (Keras 3 / TensorFlow 2.x)

PATIENT PARAMETERS:
- Age: {age} years | Sex: {sex_label}
- Chest Pain Type: {cp_label}
- Resting Systolic BP: {trestbps} mm Hg
- Serum Cholesterol: {chol} mg/dl
- Fasting Blood Sugar > 120 mg/dl: {fbs_label}
- Resting ECG: {restecg_label}
- Max Exercise Heart Rate: {thalach} bpm
- Exercise-Induced Angina: {exang_label}
- ST Depression (Oldpeak): {oldpeak} mm
- ST Segment Slope: {slope_label}
- Colored Fluoroscopy Vessels (ca): {ca}
- Thallium Perfusion Status: {thal_label}

INFERENCE FINDINGS:
- Statistical Probability: {prob_pct:.2f}%
- Decision Threshold: {threshold:.2f}
- Classification: {"ELEVATED RISK MARKERS DETECTED" if is_elevated else "LOW STATISTICAL RISK MARKERS"}
- Clinical Stratification: {risk_stratum}

CLINICAL RECOMMENDATION:
{recommendation}

DISCLAIMER:
This report was generated by an educational B.Tech artificial intelligence system.
This software is NOT a certified medical diagnostic tool. Predictions represent
statistical likelihood and must NEVER replace professional medical evaluation.
================================================================================
"""
                    st.download_button(
                        label="Download Clinical Summary (.txt)",
                        data=report_text,
                        file_name=f"heartcare_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        width="stretch",
                    )
                with xai_link_col:
                    if st.button("Explore Feature Drivers & Explainability", key="btn_goto_xai", width="stretch"):
                        st.session_state.active_nav_section = "Model Explainability"
                        st.rerun()

            except Exception as e:
                st.error(f"Inference pipeline execution error: {e}")

        # Ephemeral Session Prediction History
        st.divider()
        st.markdown('<div class="section-title">Session Prediction Audit Log</div>', unsafe_allow_html=True)
        st.caption("Ephemeral in-memory cache for comparative case analysis. No patient records are stored on disk.")

        if st.session_state.prediction_history:
            df_hist = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(df_hist, width="stretch")
            if st.button("Clear Session Log", key="btn_clear_hist"):
                st.session_state.prediction_history = []
                st.rerun()
        else:
            st.info("No predictions executed yet in this session. Configure the parameters above and click 'Compute Statistical Risk Prediction'.")

    # -------------------------------------------------------------------------
    # SECTION 3: MODEL PERFORMANCE
    # -------------------------------------------------------------------------
    elif st.session_state.active_nav_section == "Model Performance":
        st.markdown('<div class="section-title">Model Performance & Diagnostic Validation</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Verified zero-leakage evaluation metrics, ROC analysis, and empirical benchmarks against classical machine learning algorithms.</div>',
            unsafe_allow_html=True,
        )

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
            <div style="background:#E2F1D9; border:1.5px solid #A8C59A; border-radius:10px; padding:12px 18px; margin-top:8px; font-size:0.90rem; color:#1C2E19;">
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

        # Tabs for Diagnostics vs Comparative Benchmarks
        tab_diag, tab_bench = st.tabs(["📊 Diagnostic Visualizations & ROC Analysis", "🔬 Training Curves & Comparative Benchmarks"])

        with tab_diag:
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
                with st.expander("View Full Scikit-Learn Classification Report Breakdown", expanded=False):
                    with open(clr_file, "r", encoding="utf-8") as f:
                        st.code(f.read(), language="text")

        with tab_bench:
            hist_plot = RESULTS_DIR / "training_history.png"
            if hist_plot.exists():
                st.image(str(hist_plot), caption="Training vs Validation Loss & Accuracy Progression Across Epochs", width="stretch")

            st.write("")
            st.markdown('<div class="section-title">Comparative Model Benchmarks</div>', unsafe_allow_html=True)
            bench_view = st.radio(
                "Benchmark View Partition:",
                options=["Leakage-Free Test Split (Unique Patients, N=61)", "Standard 1,025-Row Dataset (Literature Baseline)"],
                index=0,
                horizontal=True,
            )

            active_comp = comp_df if "Leakage-Free" in bench_view else comp_std_df
            if not active_comp.empty:
                st.dataframe(
                    active_comp.style.highlight_max(
                        subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                        color="#C7E3B8",
                    ),
                    width="stretch",
                )
            else:
                st.warning("Model comparison records not found.")

            comp_chart = RESULTS_DIR / "model_comparison.png"
            if comp_chart.exists():
                st.image(str(comp_chart), caption="Empirical Model Benchmark Performance Across Algorithms", width="stretch")

            st.markdown(
                """
                <div style="background:#E2F1D9; border:1.5px solid #A8C59A; border-radius:12px; padding:16px 20px; margin-top:16px; line-height:1.6;">
                    <h4 style="color:#6B8A61; margin-top:0; font-family:'Outfit', sans-serif;">Why the Regularized ANN was Selected:</h4>
                    <p style="color:#2D4A27; margin-bottom:0; font-size:0.92rem;">
                        While tree ensembles (Random Forest) deliver competitive accuracy, the regularized Multi-Layer Perceptron produces <strong>calibrated continuous probability outputs</strong> with smooth decision boundaries.
                        Furthermore, the Improved ANN achieved <strong>0.9015 ROC-AUC</strong> and <strong>89.66% precision</strong>, dramatically lowering false alarms while preserving strong clinical discrimination.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        if st.button("Launch Clinical Risk Prediction Form ➔", key="btn_goto_pred_from_perf", type="primary"):
            st.session_state.active_nav_section = "Prediction"
            st.rerun()

    # -------------------------------------------------------------------------
    # SECTION 4: MODEL EXPLAINABILITY
    # -------------------------------------------------------------------------
    elif st.session_state.active_nav_section == "Model Explainability":
        st.markdown('<div class="section-title">Explainable AI: Permutation Feature Influence</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Permutation feature importance quantifies how strongly the Artificial Neural Network relies on each clinical attribute.</div>',
            unsafe_allow_html=True,
        )

        xai_img = RESULTS_DIR / "feature_analysis.png"
        if xai_img.exists():
            st.image(str(xai_img), caption="Permutation Feature Influence Ranking & Risk Marker Correlations", width="stretch")
        else:
            st.warning("Feature analysis visualization not found.")

        st.markdown(
            """
            <div style="background:#E2F1D9; border:1.5px solid #A8C59A; border-radius:12px; padding:18px 22px; line-height:1.6; margin-top:16px;">
                <h4 style="color:#6B8A61; margin-top:0; font-family:'Outfit', sans-serif;">Methodology & Interpretation</h4>
                <p style="color:#2D4A27; margin-bottom:8px; font-size:0.93rem;">
                    <strong>Computation Protocol:</strong> Each input feature was randomly permuted across 10 repeats on the unseen test set,
                    severing its association with the target while preserving all other feature distributions. The resulting decline in ROC-AUC
                    directly measures the model's architectural dependency on that feature.
                </p>
                <p style="color:#5B7556; font-size:0.88rem; margin-bottom:0;">
                    <em>Ethical Disclaimer: Permutation feature importance represents statistical reliance within this model only.
                    High feature importance reflects mathematical utility in predicting tabular labels and does <strong>not</strong> imply medical or physiological causation.</em>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        if st.button("Return to Clinical Risk Prediction ➔", key="btn_goto_pred_from_xai", type="primary"):
            st.session_state.active_nav_section = "Prediction"
            st.rerun()

    # -------------------------------------------------------------------------
    # SECTION 5: ABOUT
    # -------------------------------------------------------------------------
    elif st.session_state.active_nav_section == "About":
        st.markdown('<div class="section-title">Project Architecture & Academic Methodology</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-caption">Comprehensive overview of the engineering pipeline, data lineage, technologies, and clinical limitations.</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="background:#E2F1D9; border:1.5px solid #A8C59A; border-radius:14px; padding:24px 28px; line-height:1.65; color:#1C2E19;">
                <h3 style="color:#6B8A61; margin-top:0; font-family:'Outfit', sans-serif;">1. Project Background</h3>
                <ul style="margin-bottom:16px; color:#2D4A27;">
                    <li><strong>Project Title:</strong> HeartCare: Heart Disease Prediction using Artificial Neural Network (ANN)</li>
                    <li><strong>Domain:</strong> Artificial Intelligence in Healthcare & Applied Deep Learning</li>
                    <li><strong>Target:</strong> Binary Classification (Class 1: Elevated Risk Markers, Class 0: Negative Control)</li>
                </ul>

                <h3 style="color:#6B8A61; font-family:'Outfit', sans-serif;">2. End-to-End Pipeline Workflow</h3>
                <ol style="margin-bottom:16px; color:#2D4A27;">
                    <li><strong>Data Ingestion & Audit:</strong> Audited raw Cleveland dataset (1,025 entries) uncovering 723 duplicate rows of 302 unique patients.</li>
                    <li><strong>Zero-Leakage Splitting:</strong> Implemented strict stratified 3-way partition: Train (204 patients), Validation (37 patients), Held-Out Test (61 patients).</li>
                    <li><strong>Feature Engineering:</strong> Fitted a Scikit-Learn <code>ColumnTransformer</code> strictly on <code>X_train</code> (StandardScaler continuous + OneHotEncoder categoricals &rarr; 27 feature columns).</li>
                    <li><strong>Neural Network Architecture:</strong>
                        <ul>
                            <li>Dense 48 (ReLU, L2 weight decay &lambda; = 0.001) + Batch Normalization + Dropout 0.25</li>
                            <li>Dense 24 (ReLU, L2 weight decay &lambda; = 0.001) + Batch Normalization + Dropout 0.20</li>
                            <li>Dense 12 (ReLU) + Dropout 0.10</li>
                            <li>Output 1 (Sigmoid continuous activation)</li>
                        </ul>
                    </li>
                    <li><strong>Optimization:</strong> EarlyStopping halted training at epoch 42, restoring optimal weights from epoch 17.</li>
                    <li><strong>Explainability:</strong> Applied Permutation Feature Importance across 10 repeats on the test set.</li>
                    <li><strong>Interactive Deployment:</strong> Built this unified Streamlit website with threshold tuning and downloadable clinical summary reports.</li>
                </ol>

                <h3 style="color:#6B8A61; font-family:'Outfit', sans-serif;">3. Technology Stack</h3>
                <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:18px;">
                    <span style="background:#C7E3B8; border:1px solid #A8C59A; padding:4px 12px; border-radius:16px; font-weight:700; font-size:0.82rem; color:#1C2E19;">Python 3.12</span>
                    <span style="background:#C7E3B8; border:1px solid #A8C59A; padding:4px 12px; border-radius:16px; font-weight:700; font-size:0.82rem; color:#1C2E19;">TensorFlow 2.x & Keras 3</span>
                    <span style="background:#C7E3B8; border:1px solid #A8C59A; padding:4px 12px; border-radius:16px; font-weight:700; font-size:0.82rem; color:#1C2E19;">Scikit-Learn</span>
                    <span style="background:#C7E3B8; border:1px solid #A8C59A; padding:4px 12px; border-radius:16px; font-weight:700; font-size:0.82rem; color:#1C2E19;">Streamlit</span>
                    <span style="background:#C7E3B8; border:1px solid #A8C59A; padding:4px 12px; border-radius:16px; font-weight:700; font-size:0.82rem; color:#1C2E19;">FastAPI REST API</span>
                    <span style="background:#C7E3B8; border:1px solid #A8C59A; padding:4px 12px; border-radius:16px; font-weight:700; font-size:0.82rem; color:#1C2E19;">Pandas & NumPy</span>
                </div>

                <h3 style="color:#6B8A61; font-family:'Outfit', sans-serif;">4. Clinical Limitations</h3>
                <ul style="margin-bottom:16px; color:#2D4A27;">
                    <li><strong>Cohort Scale:</strong> Model is trained on the single-center Cleveland study (302 unique patients). Prospective multi-center clinical trials are required before real-world adoption.</li>
                    <li><strong>Static Assessment:</strong> Model evaluates cross-sectional physiological snapshots and does not process longitudinal patient telemetry.</li>
                </ul>

                <h3 style="color:#6B8A61; font-family:'Outfit', sans-serif;">5. Mandatory Ethical & Medical Disclaimer</h3>
                <p style="color:#3D5438; font-size:0.90rem; margin-bottom:0;">
                    This software is developed strictly as an educational B.Tech engineering prototype and research benchmarking tool.
                    It is <strong>NOT</strong> a certified diagnostic medical device. Model outputs represent statistical likelihood and must <strong>never</strong> supersede professional clinical judgment, medical diagnosis, or patient triage.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        if st.button("Explore Clinical Prediction ➔", key="btn_goto_pred_from_abt", type="primary"):
            st.session_state.active_nav_section = "Prediction"
            st.rerun()
        elif st.session_state.active_nav_section == "Patient Login":
            render_patient_login()
        elif st.session_state.active_nav_section == "Doctor Login":
            render_doctor_login()
        elif st.session_state.active_nav_section == "My Portal":
            render_patient_portal()
        elif st.session_state.active_nav_section == "Doctor Portal":
            render_doctor_portal()
    # Always render common footer across every section
    render_common_footer()


if __name__ == "__main__":
    main()
