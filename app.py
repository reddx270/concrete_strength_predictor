"""
app.py
-------
Streamlit app for predicting 28-day concrete compressive strength
from mix proportions. For learning/demo purposes only.
Run:  streamlit run app.py
"""
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Load model and metadata (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    main = joblib.load("model_main.joblib")
    lo = joblib.load("model_lo.joblib")
    hi = joblib.load("model_hi.joblib")
    with open("metadata.json") as f:
        meta = json.load(f)
    return main, lo, hi, meta


model_main, model_lo, model_hi, meta = load_artifacts()
ranges = meta["feature_ranges"]
metrics = meta["metrics"]

# ------------------------------------------------------------------
# Light styling
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    .big-metric {font-size: 3rem; font-weight: 700; color: #1f4e79;}
    .grade-badge {
        display:inline-block; padding:6px 14px; border-radius:20px;
        background:#1f4e79; color:white; font-weight:600; font-size:1rem;
    }
    .small-muted {color:#666; font-size:0.85rem;}
    .stSlider > div > div > div > div { background-color: #1f4e79; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("🧱 Concrete Compressive Strength Predictor")
st.markdown(
    "Predicts compressive strength of a concrete mix from its ingredient proportions, "
    "using an XGBoost model trained on ~1,600 mixes (UCI benchmark + project data + "
    "IS 10262 synthetic). **Educational/demo only — not for design use.**"
)

# ------------------------------------------------------------------
# Sidebar — model info
# ------------------------------------------------------------------
with st.sidebar:
    st.header("About the Model")
    st.metric("Test R²", f"{metrics['r2']:.3f}")
    st.metric("Test MAE", f"{metrics['mae']:.2f} MPa")
    st.metric("5-fold CV R²", f"{metrics['cv_r2_mean']:.3f}")
    st.caption(
        f"Trained on {meta['n_train']} mixes, tested on {meta['n_test']}. "
        f"Empirical coverage of 80% prediction interval on test set: "
        f"{metrics['pi_coverage_80']:.0%}."
    )
    st.divider()
    st.header("Top Drivers")
    imp_df = pd.DataFrame(meta["feature_importance"]).head(7)
    fig = go.Figure(go.Bar(
        x=imp_df["importance"][::-1],
        y=imp_df["feature"][::-1],
        orientation="h",
        marker_color="#1f4e79",
    ))
    fig.update_layout(
        height=260, margin=dict(l=0, r=0, t=10, b=10),
        xaxis_title="Importance", yaxis_title="",
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "These are the features the model relies on most. "
        "Water-to-binder ratio (WCM_Ratio) usually leads — Abrams' law in action."
    )

# ------------------------------------------------------------------
# Input section
# ------------------------------------------------------------------
st.subheader("Mix Proportions  *(per m³ of concrete)*")
st.caption(
    "Slider ranges are based on the 5th–95th percentile of the training data. "
    "Pushing toward the edges reduces prediction confidence."
)


def slider_for(col_name, label, unit, step, default_key="median", help_text=None):
    r = ranges[col_name]
    return st.slider(
        f"{label} ({unit})",
        min_value=float(round(r["min"], 1)),
        max_value=float(round(r["max"], 1)),
        value=float(round(r[default_key], 1)),
        step=step,
        help=help_text,
    )


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Binder**")
    cement = slider_for("Cement", "Cement", "kg/m³", 5.0,
                        help_text="Portland cement content. Typical structural mixes: 280–450 kg/m³.")
    ggbs = slider_for("GGBS", "GGBS (slag)", "kg/m³", 5.0,
                      default_key="min",
                      help_text="Ground granulated blast-furnace slag. Replaces cement; slows early strength but improves durability.")
    flyash = slider_for("FlyAsh", "Fly Ash", "kg/m³", 5.0,
                        default_key="min",
                        help_text="Class F or C fly ash. Lowers heat of hydration, refines pore structure.")

with col2:
    st.markdown("**Water & Admixture**")
    water = slider_for("Water", "Water", "kg/m³", 1.0,
                       help_text="Free water added to the mix. Lower water = higher strength (Abrams' law).")
    sp = slider_for("SP", "Superplasticizer", "kg/m³", 0.1,
                    default_key="min",
                    help_text="High-range water reducer. Lets you cut water without losing workability.")
    age = st.slider(
        "Age at testing (days)", 1, 365, 28, 1,
        help="28 days is the standard. Strength keeps growing with age, especially with SCMs."
    )

with col3:
    st.markdown("**Aggregates**")
    coarse = slider_for("CoarseAgg", "Coarse Aggregate", "kg/m³", 5.0,
                        help_text="Crushed stone/gravel, typically 10–20 mm nominal size.")
    fine = slider_for("FineAgg", "Fine Aggregate", "kg/m³", 5.0,
                      help_text="Sand. Acts as a filler between coarse particles.")

# ------------------------------------------------------------------
# Build feature vector
# ------------------------------------------------------------------
binder = cement + ggbs + flyash
total_kg = binder + water + sp + coarse + fine

# Defensive: avoid divide-by-zero
def safe_div(a, b):
    return a / b if b > 0 else 0.0

features = {
    "Cement": cement, "GGBS": ggbs, "FlyAsh": flyash, "Water": water,
    "SP": sp, "CoarseAgg": coarse, "FineAgg": fine, "Age": age,
    "Binder": binder,
    "WC_Ratio": safe_div(water, cement),
    "WCM_Ratio": safe_div(water, binder),
    "SCM_Pct": safe_div(ggbs + flyash, binder),
    "AggRatio": safe_div(coarse, fine),
    "SP_per_Binder": safe_div(sp, binder),
}
X_in = np.array([[features[f] for f in meta["features"]]])

# ------------------------------------------------------------------
# Predict
# ------------------------------------------------------------------
y_pred = float(model_main.predict(X_in)[0])
y_lo = float(model_lo.predict(X_in)[0])
y_hi = float(model_hi.predict(X_in)[0])
y_lo, y_hi = min(y_lo, y_pred), max(y_hi, y_pred)  # guard against crossing

# IS 456 nearest grade (just informational)
IS_GRADES = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
nearest_grade = max((g for g in IS_GRADES if g <= y_pred), default=IS_GRADES[0])

# ------------------------------------------------------------------
# Results
# ------------------------------------------------------------------
st.divider()
res_l, res_r = st.columns([1, 1])

with res_l:
    st.subheader("Predicted Compressive Strength")
    st.markdown(
        f"<div class='big-metric'>{y_pred:.1f} <span style='font-size:1.5rem;'>MPa</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**80% prediction interval:** {y_lo:.1f} – {y_hi:.1f} MPa  \n"
        f"<span class='small-muted'>The true strength is likely (with ~80% probability, "
        f"based on the test set) to fall in this band.</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"&nbsp;&nbsp;&nbsp;<span class='grade-badge'>Closest IS 456 grade: M{nearest_grade}</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Note: M-grade is based on characteristic 28-day strength. This is a quick "
        "reference, not a design statement."
    )

with res_r:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[y_lo, y_hi], y=[1, 1], mode="lines",
        line=dict(width=14, color="#9ec7e7"),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[y_pred], y=[1], mode="markers",
        marker=dict(size=22, color="#1f4e79", line=dict(color="white", width=2)),
        showlegend=False,
        hovertemplate=f"Predicted: {y_pred:.1f} MPa<extra></extra>"))
    fig.add_annotation(x=y_lo, y=1.25, text=f"{y_lo:.1f}", showarrow=False, font=dict(size=12))
    fig.add_annotation(x=y_hi, y=1.25, text=f"{y_hi:.1f}", showarrow=False, font=dict(size=12))
    fig.add_annotation(x=y_pred, y=0.6, text=f"<b>{y_pred:.1f} MPa</b>",
                       showarrow=False, font=dict(size=14, color="#1f4e79"))
    fig.update_layout(
        height=180, margin=dict(l=20, r=20, t=10, b=10),
        xaxis=dict(range=[0, max(100, y_hi + 10)], title="Strength (MPa)"),
        yaxis=dict(visible=False, range=[0, 2]),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Mix derived properties (engineer's eye)
# ------------------------------------------------------------------
st.divider()
st.subheader("Mix Diagnostics")

dcol1, dcol2, dcol3, dcol4 = st.columns(4)
dcol1.metric("Total Binder", f"{binder:.0f} kg/m³")
dcol2.metric("w/c Ratio", f"{features['WC_Ratio']:.2f}")
dcol3.metric("w/cm Ratio", f"{features['WCM_Ratio']:.2f}")
dcol4.metric("SCM Replacement", f"{features['SCM_Pct']*100:.0f}%")

dcol5, dcol6, dcol7, dcol8 = st.columns(4)
dcol5.metric("Coarse/Fine Ratio", f"{features['AggRatio']:.2f}")
dcol6.metric("SP / Binder", f"{features['SP_per_Binder']*100:.2f}%")
dcol7.metric("Total Mass", f"{total_kg:.0f} kg/m³")

# Density sanity flag
if total_kg < 2200:
    dcol8.metric("Density Check", "⚠️ Low",
                 help="Below typical 2200–2500 kg/m³ — air voids implied or mix is incomplete.")
elif total_kg > 2500:
    dcol8.metric("Density Check", "⚠️ High",
                 help="Above typical range — check proportions.")
else:
    dcol8.metric("Density Check", "✅ OK")

# Engineering sanity warnings
warnings = []
if features["WCM_Ratio"] > 0.65:
    warnings.append("**w/cm > 0.65** — strength will be low and durability poor (IS 456 max for moderate exposure: 0.50).")
if features["WCM_Ratio"] < 0.28:
    warnings.append("**w/cm < 0.28** — extremely low; needs heavy SP and may not be workable in practice.")
if binder > 540:
    warnings.append("**Binder > 540 kg/m³** — IS 456 caps total cementitious at 540 kg/m³ to limit shrinkage and heat.")
if binder < 250 and y_pred < 25:
    warnings.append("**Low binder content** — strength likely too low for structural concrete.")
if features["SCM_Pct"] > 0.7:
    warnings.append("**SCM replacement > 70%** — outside typical practice; early-age strength will be very low.")

if warnings:
    with st.expander("⚠️ Engineering Sanity Notes", expanded=True):
        for w in warnings:
            st.markdown(f"- {w}")

# ------------------------------------------------------------------
# Honesty section
# ------------------------------------------------------------------
st.divider()
with st.expander("📘 What this model can and can't do (read me)"):
    st.markdown(
        """
**What it does well**
- Captures the dominant relationships in concrete strength: water-to-binder ratio, total binder, age, and SCM substitution.
- 5-fold CV R² of {r2:.2f} on a heterogeneous dataset.

**What it does *not* know about**
- **Cement type/grade** (OPC 43 vs OPC 53 vs PPC vs PSC) — treated as one number.
- **Aggregate properties**: specific gravity, gradation, max size, water absorption, shape.
- **Curing conditions**, mixing time, transport time, ambient temperature, humidity.
- **Admixture chemistry** — all superplasticizers are treated identically.
- **Air content** is not an input.

**Data caveats you should know**
- ~62% of the training data is the UCI benchmark (largely US/Taiwan origin, varied ages).
- ~7% is real Mumbai project data (limited to ages 1–28 days).
- ~30% is synthetic data generated to follow IS 10262 design relationships — useful for breadth but learning from a formula, not from the real world.
- The model is best at predicting **28-day strength** for mixes within the typical range. It extrapolates poorly.

**Bottom line**
This is a learning project. Real mix design uses IS 10262 with target mean strength,
trial mixes, and lab verification. No predictor — including this one — replaces that.
        """.format(r2=metrics["cv_r2_mean"])
    )

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown(
    "<br><div class='small-muted' style='text-align:center;'>"
    "Built with Streamlit, pandas, scikit-learn and XGBoost · "
    "Dataset: UCI Concrete Compressive Strength + project data + IS 10262 synthetic"
    "</div>", unsafe_allow_html=True
)
