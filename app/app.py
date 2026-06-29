"""
MCI Conversion Risk Predictor — Streamlit App

Run: streamlit run app/app.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MCI Conversion Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Hero banner */
  .hero-banner {
    background: linear-gradient(135deg, #0f2340 0%, #1e3a5f 50%, #0d5f54 100%);
    border-radius: 14px;
    padding: 2rem 2.2rem 1.6rem;
    margin-bottom: 1.8rem;
    color: white;
  }
  .hero-title {
    font-size: 2rem; font-weight: 800; letter-spacing: -0.03em;
    margin: 0 0 0.3rem;
  }
  .hero-subtitle {
    font-size: 0.97rem; opacity: 0.82; line-height: 1.55; margin: 0;
    max-width: 620px;
  }
  .hero-badges { margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
  .badge {
    background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.25);
    color: white; font-size: 0.72rem; font-weight: 600; padding: 0.25rem 0.7rem;
    border-radius: 20px; letter-spacing: 0.04em;
  }

  /* Section headers with colored left accent */
  .section-label {
    display: flex; align-items: center; gap: 0.5rem;
    font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 0.6rem; margin-top: 0.2rem;
  }
  .section-label.blue  { color: #1d4ed8; }
  .section-label.teal  { color: #0d9488; }
  .section-label.violet{ color: #7c3aed; }
  .section-label.green { color: #15803d; }
  .dot {
    width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
  }
  .dot.blue  { background: #3b82f6; }
  .dot.teal  { background: #14b8a6; }
  .dot.violet{ background: #8b5cf6; }
  .dot.green { background: #22c55e; }

  /* Input cards */
  .input-card {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 1rem 1.1rem 0.6rem; margin-bottom: 0.5rem;
  }
  .input-card.blue-card   { border-top: 3px solid #3b82f6; }
  .input-card.teal-card   { border-top: 3px solid #14b8a6; }
  .input-card.violet-card { border-top: 3px solid #8b5cf6; }
  .input-card.green-card  { border-top: 3px solid #22c55e; }

  /* Result banner */
  .result-banner-high {
    background: linear-gradient(135deg, #7f1d1d, #dc2626);
    border-radius: 12px; padding: 1.6rem 2rem; color: white; text-align: center;
    margin-bottom: 1rem;
  }
  .result-banner-moderate {
    background: linear-gradient(135deg, #78350f, #d97706);
    border-radius: 12px; padding: 1.6rem 2rem; color: white; text-align: center;
    margin-bottom: 1rem;
  }
  .result-banner-low {
    background: linear-gradient(135deg, #14532d, #16a34a);
    border-radius: 12px; padding: 1.6rem 2rem; color: white; text-align: center;
    margin-bottom: 1rem;
  }
  .result-pct {
    font-size: 3.8rem; font-weight: 900; line-height: 1; letter-spacing: -0.04em;
  }
  .result-label-text {
    font-size: 0.92rem; opacity: 0.88; margin-top: 0.3rem;
  }
  .result-band {
    font-size: 1.1rem; font-weight: 800; letter-spacing: 0.08em;
    text-transform: uppercase; margin-top: 0.8rem;
    background: rgba(255,255,255,0.2); border-radius: 6px;
    display: inline-block; padding: 0.3rem 1.2rem;
  }
  .result-desc { font-size: 0.87rem; opacity: 0.85; margin-top: 0.5rem; }

  /* Model badge */
  .model-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 20px;
    padding: 0.3rem 0.9rem; font-size: 0.78rem; font-weight: 600; color: #1d4ed8;
  }
  .model-badge.teal-badge {
    background: #f0fdfa; border-color: #99f6e4; color: #0d9488;
  }

  /* Info pills */
  .info-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.7rem 0; }
  .info-pill {
    background: white; border: 1px solid #e5e7eb; border-radius: 8px;
    padding: 0.4rem 0.8rem; font-size: 0.78rem; color: #374151;
  }
  .info-pill b { color: #1e3a5f; }

  /* SHAP section */
  .shap-header {
    font-size: 0.95rem; font-weight: 700; color: #1e3a5f; margin-bottom: 0.1rem;
  }
  .shap-caption {
    font-size: 0.78rem; color: #6b7280; margin-bottom: 0.7rem; line-height: 1.5;
  }

  /* Sidebar — clean clinical white */
  div[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
  }
  div[data-testid="stSidebar"] * { color: #1f2937 !important; }
  div[data-testid="stSidebar"] h3 {
    color: #0f2340 !important; font-weight: 800 !important;
    letter-spacing: -0.01em; font-size: 1rem !important;
  }
  div[data-testid="stSidebar"] a { color: #0d9488 !important; font-weight: 500 !important; }
  div[data-testid="stSidebar"] hr { border-color: #e2e8f0 !important; }
  div[data-testid="stSidebar"] p,
  div[data-testid="stSidebar"] span { color: #374151 !important; }
  div[data-testid="stSidebar"] .stCheckbox label { color: #374151 !important; }
  div[data-testid="stSidebar"] .stCheckbox { background: #f8fafc; border-radius: 6px; padding: 0.2rem 0.4rem; }

  /* Sidebar top accent strip */
  div[data-testid="stSidebar"]::before {
    content: ""; display: block; height: 4px;
    background: linear-gradient(90deg, #1e3a5f, #0d9488);
  }

  .sb-metric {
    background: #f8fafc; border: 1px solid #e2e8f0; border-left: 3px solid #0d9488;
    border-radius: 8px; padding: 0.6rem 0.85rem; margin-bottom: 0.5rem;
  }
  .sb-metric-label {
    font-size: 0.67rem; color: #6b7280 !important; text-transform: uppercase;
    letter-spacing: 0.09em; font-weight: 600;
  }
  .sb-metric-value {
    font-size: 1.05rem; font-weight: 800; color: #1e3a5f !important;
    margin-top: 0.2rem; letter-spacing: -0.01em;
  }

  /* Disclaimer */
  .disclaimer {
    background: #fefce8; border: 1px solid #fde047; border-radius: 8px;
    padding: 0.8rem 1rem; font-size: 0.8rem; color: #713f12; margin-top: 0.5rem;
  }
  .footer-bar {
    text-align: center; font-size: 0.78rem; color: #9ca3af; margin-top: 0.5rem;
  }
  .footer-bar a { color: #0d9488 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "MMSCORE":              "MMSE Score",
    "CDRSB":                "CDR Sum of Boxes",
    "FAQTOTAL":             "FAQ Total",
    "GDTOTAL":              "GDS (Depression)",
    "CATANIMSC":            "Animal Fluency",
    "LIMMTOTAL":            "Logical Memory Immed.",
    "LDELTOTAL":            "Logical Memory Delayed",
    "TRAASCOR":             "Trail Making A",
    "TRABSCOR":             "Trail Making B",
    "PTEDUCAT":             "Years of Education",
    "AGE":                  "Age",
    "APOE4_count":          "APOE ε4 Alleles",
    "sex_female":           "Female Sex",
    "race_nonwhite":        "Non-white Race",
    "ethnicity_hispanic":   "Hispanic Ethnicity",
    "married":              "Married",
    "TOTSCORE":             "ADAS-11 Score",
    "TOTAL13":              "ADAS-13 Score",
    "RAVLT_learning_total": "RAVLT Learning",
    "AVDEL30MIN":           "RAVLT 30-min Delay",
    "AVDELTOT":             "RAVLT Total Delay",
    "Hippocampus_total":    "Hippocampus Vol.",
    "Entorhinal_total":     "Entorhinal Vol.",
    "Ventricle_total":      "Ventricle Vol.",
    "Fusiform_total":       "Fusiform Vol.",
    "ICV":                  "Intracranial Vol.",
}


# ── Load bundle ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_bundle():
    for path in [
        "models/mci_conversion_predictor.joblib",
        "../models/mci_conversion_predictor.joblib",
        os.path.join(os.path.dirname(__file__), "../models/mci_conversion_predictor.joblib"),
    ]:
        if os.path.exists(path):
            return joblib.load(path)
    return None


# ── Charts ─────────────────────────────────────────────────────────────────────
def plot_gauge(probability):
    """Semicircular risk gauge with zone coloring and needle."""
    if probability >= 0.66:
        needle_color = "#dc2626"
    elif probability >= 0.33:
        needle_color = "#d97706"
    else:
        needle_color = "#16a34a"

    fig, ax = plt.subplots(figsize=(5, 3.2), facecolor="white")
    ax.set_facecolor("white")

    # Zone arcs (background track)
    kw = dict(linewidth=26, solid_capstyle="butt")
    ax.plot(*zip(*[(np.cos(t), np.sin(t)) for t in np.linspace(np.pi, np.pi * 0.67, 300)]),
            color="#dcfce7", **kw)
    ax.plot(*zip(*[(np.cos(t), np.sin(t)) for t in np.linspace(np.pi * 0.67, np.pi * 0.34, 300)]),
            color="#fef9c3", **kw)
    ax.plot(*zip(*[(np.cos(t), np.sin(t)) for t in np.linspace(np.pi * 0.34, 0, 300)]),
            color="#fee2e2", **kw)

    # Zone dividers
    for frac in [0.33, 0.66]:
        angle = np.pi * (1 - frac)
        ax.plot([np.cos(angle) * 0.73, np.cos(angle) * 1.02],
                [np.sin(angle) * 0.73, np.sin(angle) * 1.02],
                color="white", linewidth=3.5, zorder=6)

    # Filled arc up to probability
    prob_angle = np.pi * (1 - probability)
    pts = [(np.cos(t), np.sin(t)) for t in np.linspace(np.pi, prob_angle, 400)]
    ax.plot(*zip(*pts), color=needle_color, linewidth=14,
            solid_capstyle="round", zorder=4, alpha=0.92)

    # Needle
    ax.annotate(
        "",
        xy=(np.cos(prob_angle) * 0.62, np.sin(prob_angle) * 0.62),
        xytext=(0, 0),
        arrowprops=dict(
            arrowstyle="-|>", color="#1e3a5f", lw=2.0,
            mutation_scale=12,
        ),
        zorder=8,
    )
    # Center hub
    hub = plt.Circle((0, 0), 0.065, color="#1e3a5f", zorder=9)
    ax.add_patch(hub)

    # Probability text
    ax.text(0, 0.28, f"{probability * 100:.1f}%", ha="center", va="center",
            fontsize=21, fontweight="900", color=needle_color)
    ax.text(0, 0.06, "24-month risk", ha="center", va="center",
            fontsize=8, color="#6b7280")

    # Zone labels
    ax.text(-1.18, 0.08, "LOW", ha="center", fontsize=8, color="#16a34a", fontweight="700")
    ax.text(0, 1.22, "MED", ha="center", fontsize=8, color="#d97706", fontweight="700")
    ax.text(1.18, 0.08, "HIGH", ha="center", fontsize=8, color="#dc2626", fontweight="700")

    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-0.35, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=0.3)
    return fig


def plot_shap_bar(shap_vals, feature_names, top_n=12):
    """Styled horizontal bar chart of top SHAP contributions."""
    df = pd.DataFrame({"feature": feature_names, "shap": shap_vals})
    df["abs"] = df["shap"].abs()
    df = df.nlargest(top_n, "abs").sort_values("shap")

    labels = [FEATURE_LABELS.get(f, f) for f in df["feature"]]
    values = df["shap"].values
    colors = ["#ef4444" if v > 0 else "#0d9488" for v in values]

    fig, ax = plt.subplots(figsize=(7.5, max(4.5, len(df) * 0.52)))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    bars = ax.barh(labels, values, color=colors, edgecolor="none", height=0.62, alpha=0.88)

    # Value labels
    for bar, val in zip(bars, values):
        w = bar.get_width()
        x_text = w + 0.003 if w >= 0 else w - 0.003
        ha = "left" if w >= 0 else "right"
        ax.text(x_text, bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", ha=ha, fontsize=7.5, color="#374151")

    ax.axvline(0, color="#94a3b8", linewidth=1.0, zorder=3)
    ax.set_xlabel("← Reduces risk     |     Increases risk →", fontsize=8, color="#6b7280")
    ax.set_title("What Drove This Prediction", fontsize=11,
                 fontweight="bold", color="#1e3a5f", pad=10)
    ax.tick_params(axis="y", labelsize=8.5, colors="#374151")
    ax.tick_params(axis="x", labelsize=7.5, colors="#94a3b8")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#e2e8f0")

    legend_els = [
        mpatches.Patch(color="#ef4444", label="↑ Increases conversion risk"),
        mpatches.Patch(color="#0d9488", label="↓ Decreases conversion risk"),
    ]
    ax.legend(handles=legend_els, fontsize=7.5, loc="lower right",
              frameon=True, framealpha=0.95, edgecolor="#e2e8f0")

    plt.tight_layout(pad=0.8)
    return fig


# ── SHAP values ────────────────────────────────────────────────────────────────
def compute_shap(model, features, patient_values):
    try:
        import shap
        imputer    = model.named_steps["impute"]
        scaler     = model.named_steps["scale"]
        classifier = model.named_steps["model"]
        X = np.array([patient_values])
        X_sca = scaler.transform(imputer.transform(X))
        background = np.zeros((1, X_sca.shape[1]))
        explainer = shap.LinearExplainer(
            classifier, background, feature_perturbation="interventional"
        )
        sv = explainer.shap_values(X_sca)
        if isinstance(sv, list):
            sv = sv[1]
        return sv.flatten()
    except Exception:
        try:
            imputer    = model.named_steps["impute"]
            scaler     = model.named_steps["scale"]
            classifier = model.named_steps["model"]
            X = np.array([patient_values])
            X_sca = scaler.transform(imputer.transform(X))
            return classifier.coef_[0] * X_sca[0]
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧠 MCI Risk Predictor")
    st.markdown(
        "Predicts **24-month dementia conversion probability** from baseline MCI data. "
        "Trained on ADNI · Validated on NACC."
    )
    st.markdown("---")

    st.markdown("**Model Performance**")
    st.markdown(
        '<div class="sb-metric">'
        '<div class="sb-metric-label">Primary Model AUC (ADNI)</div>'
        '<div class="sb-metric-value">0.887</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sb-metric">'
        '<div class="sb-metric-label">Accessible Model AUC (ADNI)</div>'
        '<div class="sb-metric-value">0.871</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sb-metric">'
        '<div class="sb-metric-label">External Validation (NACC, n=1,974)</div>'
        '<div class="sb-metric-value">0.805 [0.784, 0.826]</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sb-metric">'
        '<div class="sb-metric-label">Training Cohort</div>'
        '<div class="sb-metric-value">816 ADNI patients</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    use_recal = st.checkbox(
        "Use recalibrated threshold (0.261)",
        value=False,
        help=(
            "Default threshold = 0.5. The recalibrated value of 0.261 was tuned on "
            "NACC using Youden's J and improves converter recall from 0.63 → 0.783 "
            "in diverse populations, at the cost of lower precision."
        ),
    )
    st.markdown("---")
    st.markdown(
        "**Victor C. Onuh**\n\n"
        "[GitHub](https://github.com/Vichizzy) · "
        "[LinkedIn](https://www.linkedin.com/in/victor-c-onuh-81647b208)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">🧠 MCI Conversion Risk Predictor</p>
  <p class="hero-subtitle">
    Enter a patient's baseline clinical measurements to estimate their probability of
    converting from Mild Cognitive Impairment to dementia within 24 months.
    Missing fields are automatically imputed — enter only what you have.
  </p>
  <div class="hero-badges">
    <span class="badge">ADNI Trained</span>
    <span class="badge">NACC Validated</span>
    <span class="badge">AUC 0.887</span>
    <span class="badge">SHAP Explainable</span>
    <span class="badge">Research Tool</span>
  </div>
</div>
""", unsafe_allow_html=True)

bundle = load_bundle()
if bundle is None:
    st.error(
        "**Model bundle not found.** Place `mci_conversion_predictor.joblib` in `models/`. "
        "See `models/README.md`."
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# INPUT FORM — tabbed layout
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("#### Enter Patient Data")
st.caption(
    "The model **auto-selects** between the 26-feature and 16-feature model based on "
    "what you provide. Fill in any combination — the model handles the rest."
)

patient = {}

tab1, tab2, tab3 = st.tabs([
    "🧠  Cognitive & Neuropsych  (always used)",
    "🏥  Advanced Tests  (unlocks full model)",
    "👤  Demographics & Genetics",
])

# ── Tab 1: Cognitive & Neuropsych ──────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="section-label blue"><div class="dot blue"></div>Cognitive Assessment</div>',
            unsafe_allow_html=True,
        )
        patient["MMSCORE"] = st.number_input(
            "MMSE Score (0–30, higher = better)", 0, 30, value=None,
            help="Mini-Mental State Examination. Most widely used cognitive screen globally.",
        )
        patient["CDRSB"] = st.number_input(
            "CDR Sum of Boxes (0–18)", 0.0, 18.0, value=None, step=0.5,
            help="Clinical Dementia Rating Sum of Boxes. 0.5 global CDR defines MCI in ADNI.",
        )
        patient["FAQTOTAL"] = st.number_input(
            "FAQ Total (0–30)", 0, 30, value=None,
            help="Functional Activities Questionnaire. Scores >9 indicate functional impairment.",
        )
        patient["GDTOTAL"] = st.number_input(
            "GDS Total — Depression (0–15)", 0, 15, value=None,
            help="Geriatric Depression Scale. Depression is a conversion cofactor.",
        )

    with c2:
        st.markdown(
            '<div class="section-label teal"><div class="dot teal"></div>Neuropsychological Tests</div>',
            unsafe_allow_html=True,
        )
        patient["CATANIMSC"] = st.number_input(
            "Animal Fluency (count/60 sec)", 0, 80, value=None,
            help="Animals named in 60 seconds. Tests semantic memory and language.",
        )
        patient["LIMMTOTAL"] = st.number_input(
            "Logical Memory — Immediate (0–25)", 0, 25, value=None,
            help="Story recall immediately after hearing. Part of WMS-R.",
        )
        patient["LDELTOTAL"] = st.number_input(
            "Logical Memory — Delayed (0–25)", 0, 25, value=None,
            help="Story recall after a 30-minute delay. Strong AD conversion signal.",
        )
        patient["TRAASCOR"] = st.number_input(
            "Trail Making Test A (seconds)", 0, 300, value=None,
            help="Connect numbered circles in order. Processing speed. Lower = better.",
        )
        patient["TRABSCOR"] = st.number_input(
            "Trail Making Test B (seconds)", 0, 600, value=None,
            help="Alternate numbers and letters. Executive function. Lower = better.",
        )

# ── Tab 2: Advanced Tests ──────────────────────────────────────────────────────
with tab2:
    st.info(
        "**Optional.** Providing **5 or more** of these 10 inputs triggers the "
        "**26-feature model** (AUC 0.887 vs 0.871). Leave blank if unavailable.",
        icon="ℹ️",
    )
    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown(
            '<div class="section-label violet"><div class="dot violet"></div>ADAS Battery</div>',
            unsafe_allow_html=True,
        )
        patient["TOTSCORE"] = st.number_input(
            "ADAS-11 Score (0–70, higher = worse)", 0, 70, value=None,
            help="Alzheimer's Disease Assessment Scale — 11 items.",
        )
        patient["TOTAL13"] = st.number_input(
            "ADAS-13 Score (0–85, higher = worse)", 0, 85, value=None,
            help="ADAS-13 adds delayed recall + number cancellation. More MCI-sensitive.",
        )

    with a2:
        st.markdown(
            '<div class="section-label violet"><div class="dot violet"></div>RAVLT Memory</div>',
            unsafe_allow_html=True,
        )
        patient["RAVLT_learning_total"] = st.number_input(
            "RAVLT Learning Total (0–75)", 0, 75, value=None,
            help="Sum across 5 learning trials. The strongest single predictor (SHAP 0.728).",
        )
        patient["AVDEL30MIN"] = st.number_input(
            "RAVLT 30-min Delay (0–15)", 0, 15, value=None,
        )
        patient["AVDELTOT"] = st.number_input(
            "RAVLT Total Delay (0–15)", 0, 15, value=None,
        )

    with a3:
        st.markdown(
            '<div class="section-label violet"><div class="dot violet"></div>MRI Brain Volumes (mm³)</div>',
            unsafe_allow_html=True,
        )
        patient["Hippocampus_total"] = st.number_input(
            "Hippocampus (mm³)", 1000, 12000, value=None,
            help="Left + right hippocampal volume. Typical: 5,500–9,000 mm³.",
        )
        patient["Entorhinal_total"] = st.number_input(
            "Entorhinal Cortex (mm³)", 500, 8000, value=None,
            help="First region affected by Alzheimer's pathology.",
        )
        patient["Ventricle_total"] = st.number_input(
            "Lateral Ventricles (mm³)", 5000, 100000, value=None,
            help="Enlarges as brain tissue is lost in AD.",
        )
        patient["Fusiform_total"] = st.number_input(
            "Fusiform Gyrus (mm³)", 5000, 50000, value=None,
        )
        patient["ICV"] = st.number_input(
            "Intracranial Volume (mm³)", 800000, 2500000, value=None,
            help="Used to normalize regional volumes.",
        )

# ── Tab 3: Demographics ────────────────────────────────────────────────────────
with tab3:
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(
            '<div class="section-label green"><div class="dot green"></div>Patient Profile</div>',
            unsafe_allow_html=True,
        )
        patient["AGE"] = st.number_input("Age at baseline (years)", 50, 100, value=None)
        patient["PTEDUCAT"] = st.number_input("Years of education", 0, 30, value=None)
        patient["APOE4_count"] = st.selectbox(
            "APOE ε4 alleles (0, 1, or 2)",
            options=[None, 0, 1, 2],
            format_func=lambda x: "Select…" if x is None else str(x),
            help="Each APOE ε4 allele significantly increases late-onset AD risk.",
        )

    with d2:
        st.markdown(
            '<div class="section-label green"><div class="dot green"></div>Demographics</div>',
            unsafe_allow_html=True,
        )
        patient["sex_female"] = st.selectbox(
            "Sex",
            options=[None, 0, 1],
            format_func=lambda x: "Select…" if x is None else ("Female" if x == 1 else "Male"),
        )
        patient["race_nonwhite"] = st.selectbox(
            "Race",
            options=[None, 0, 1],
            format_func=lambda x: "Select…" if x is None else ("Non-white" if x == 1 else "White"),
        )
        patient["ethnicity_hispanic"] = st.selectbox(
            "Ethnicity",
            options=[None, 0, 1],
            format_func=lambda x: "Select…" if x is None else ("Hispanic" if x == 1 else "Non-Hispanic"),
        )
        patient["married"] = st.selectbox(
            "Marital status",
            options=[None, 0, 1],
            format_func=lambda x: "Select…" if x is None else ("Married" if x == 1 else "Not married"),
        )


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT BUTTON
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("")
predict_btn = st.button("🔮  Predict Conversion Risk", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if predict_btn:

    # Clean inputs
    clean = {k: (np.nan if v is None else v) for k, v in patient.items()}

    full_only = bundle["full_only_features"]
    premium = [
        f for f in full_only
        if f in clean and not (isinstance(clean[f], float) and np.isnan(clean[f]))
    ]
    use_full = len(premium) >= (len(full_only) / 2)

    if use_full:
        model       = bundle["full_model"]
        features    = bundle["full_features"]
        model_name  = "26-feature model"
        model_auc   = bundle["performance"]["full_model_adni_cv_auc"]
        badge_class = "model-badge"
        badge_icon  = "⚕️"
    else:
        model       = bundle["reduced_model"]
        features    = bundle["reduced_features"]
        model_name  = "16-feature accessible model"
        model_auc   = bundle["performance"]["reduced_model_adni_cv_auc"]
        badge_class = "model-badge teal-badge"
        badge_icon  = "✅"

    X_row = {f: clean.get(f, np.nan) for f in features}
    X_df  = pd.DataFrame([X_row])[features]

    probability = float(model.predict_proba(X_df)[0, 1])
    threshold   = bundle["nacc_recalibrated_threshold"] if use_recal else bundle["default_threshold"]
    predicted   = int(probability >= threshold)
    pct         = probability * 100

    provided = [f for f in features if not (isinstance(X_row.get(f), float) and np.isnan(X_row.get(f, np.nan)))]
    imputed  = [f for f in features if f not in provided]

    # Risk level
    if probability >= 0.66:
        banner_cls = "result-banner-high"
        risk_band  = "HIGH RISK"
        risk_icon  = "🔴"
        risk_desc  = "Close monitoring and early intervention planning are recommended."
    elif probability >= 0.33:
        banner_cls = "result-banner-moderate"
        risk_band  = "MODERATE RISK"
        risk_icon  = "🟡"
        risk_desc  = "Warrants attentive follow-up and re-evaluation within 12 months."
    else:
        banner_cls = "result-banner-low"
        risk_band  = "LOWER RISK"
        risk_icon  = "🟢"
        risk_desc  = "Routine monitoring within the 24-month window is appropriate."

    st.markdown("---")
    st.markdown("## Prediction Result")

    # ── Model routing badge ─────────────────────────────────────────────────────
    st.markdown(
        f'<div style="margin-bottom:1rem;">'
        f'<span class="{badge_class}">{badge_icon} {model_name}  ·  AUC {model_auc}</span>'
        f'&nbsp;&nbsp;<span style="font-size:0.82rem;color:#6b7280;">'
        f'{len(premium)}/{len(full_only)} advanced features provided · '
        f'{len(imputed)} features imputed</span></div>',
        unsafe_allow_html=True,
    )

    # ── Main results layout ─────────────────────────────────────────────────────
    left, mid, right = st.columns([1.1, 1, 1.6])

    with left:
        # Risk banner
        st.markdown(
            f'<div class="{banner_cls}">'
            f'<div class="result-pct">{pct:.1f}%</div>'
            f'<div class="result-label-text">probability of conversion<br>to dementia within 24 months</div>'
            f'<div class="result-band">{risk_icon} {risk_band}</div>'
            f'<div class="result-desc">{risk_desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        threshold_label = f"Recalibrated (NACC): {threshold:.3f}" if use_recal else f"Default: {threshold:.1f}"
        st.caption(f"Decision threshold — {threshold_label} → Predicted: {'**Converter**' if predicted else '**Stable**'}")

    with mid:
        # Gauge chart
        st.pyplot(plot_gauge(probability), use_container_width=True)

        # Quick metrics
        m1, m2 = st.columns(2)
        m1.metric("Stable probability", f"{(1 - probability) * 100:.1f}%")
        m2.metric("Features measured", f"{len(provided)}/{len(features)}")

        if imputed:
            with st.expander(f"ℹ️ {len(imputed)} features were imputed"):
                st.caption("Filled with training-data medians:")
                for f in imputed:
                    st.markdown(f"- {FEATURE_LABELS.get(f, f)}")

    with right:
        # SHAP chart
        st.markdown('<p class="shap-header">Feature Contributions (SHAP)</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="shap-caption">Each bar shows how much a feature shifted this '
            'prediction away from the median training patient.<br>'
            '<span style="color:#ef4444;font-weight:600;">Red → increases risk</span> · '
            '<span style="color:#0d9488;font-weight:600;">Teal → decreases risk</span></p>',
            unsafe_allow_html=True,
        )
        patient_vals = [X_row.get(f, np.nan) for f in features]
        shap_vals = compute_shap(model, features, patient_vals)

        if shap_vals is not None:
            st.pyplot(plot_shap_bar(shap_vals, features), use_container_width=True)
        else:
            st.warning("Install `shap` to enable feature explanations: `pip install shap`")

    # ── Performance context ─────────────────────────────────────────────────────
    with st.expander("📊 Full model performance context"):
        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown("""
**ADNI Training Results**

| Model | AUC (CV) | 95% CI | Recall |
|---|---|---|---|
| LR — 26 features | **0.887** | [0.832, 0.939] | 0.75 |
| RF — 26 features | 0.891 | [0.836, 0.955] | 0.50 |
| LR + CSF | 0.893\* | [0.836, 0.955] | — |
| LR — 16 features | **0.871** | — | — |

\*Not significantly different from primary model.
""")
        with pc2:
            st.markdown("""
**NACC External Validation**

| Subgroup | AUC |
|---|---|
| Overall (n=1,974) | **0.805** [0.784, 0.826] |
| White | 0.798 [0.775, 0.822] |
| Non-white | 0.832 [0.781, 0.880] |
| Male | 0.810 |
| Female | 0.801 |
| Age < 73 | 0.842 |
| Age ≥ 73 | 0.771 |

No racial or sex disparity detected.
""")

# ── Disclaimer & Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="disclaimer">'
    "⚠️ <strong>Research & Educational Use Only.</strong> This tool has not been "
    "prospectively validated for clinical use. Predictions must not guide patient care "
    "or treatment decisions without qualified clinical oversight. All estimates reflect "
    "population-level patterns and carry uncertainty — see the project README for limitations."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="footer-bar">Victor C. Onuh · '
    '<a href="https://github.com/Vichizzy/Alzheimer-MCI-prediction">GitHub</a> · '
    'Trained on ADNI · Validated on NACC</div>',
    unsafe_allow_html=True,
)
