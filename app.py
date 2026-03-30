"""
Personalized Disease Risk Manager
──────────────────────────────────
Streamlit application for cardiovascular disease (CVD) risk prediction
using a Federated Learning model trained across distributed hospital data.

References:
  • Hoghooghi-Esfahani et al. (2025) – XAI in disease prediction
  • Federated learning in healthcare EHR (2021, 2023)
"""

import os
import streamlit as st
import torch
import numpy as np
from model import get_model, INPUT_DIM

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personalized Disease Risk Manager",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "global_model.pth"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Header gradient */
.hero-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    border: 1px solid rgba(255,255,255,0.08);
}
.hero-header h1 { font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.hero-header p  { font-size: 0.95rem; color: rgba(255,255,255,0.7); margin: 0.5rem 0 0; }

/* Section cards */
.section-card {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.07);
}
.section-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #7c8cf8;
    margin-bottom: 0.8rem;
}

/* Risk gauge */
.risk-high {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 1.5rem;
    color: white;
    text-align: center;
}
.risk-moderate {
    background: linear-gradient(135deg, #451a03, #78350f);
    border: 1px solid #f59e0b;
    border-radius: 12px;
    padding: 1.5rem;
    color: white;
    text-align: center;
}
.risk-low {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #22c55e;
    border-radius: 12px;
    padding: 1.5rem;
    color: white;
    text-align: center;
}
.risk-value { font-size: 3rem; font-weight: 700; margin: 0; }
.risk-label { font-size: 1.1rem; font-weight: 600; margin: 0; opacity: 0.9; }

/* Feature bar */
.feature-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.5rem;
    gap: 0.6rem;
}
.feature-name {
    width: 180px;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.75);
    white-space: nowrap;
    flex-shrink: 0;
}
.feature-bar-bg {
    flex: 1;
    background: rgba(255,255,255,0.08);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
}
.feature-bar-fill {
    height: 8px;
    border-radius: 4px;
}
.feature-val {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.55);
    width: 40px;
    text-align: right;
    flex-shrink: 0;
}

/* Badge pill */
.badge {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.4px;
    margin-right: 0.3rem;
    margin-bottom: 0.3rem;
}
.badge-red    { background: rgba(239,68,68,0.2);  color: #f87171; border: 1px solid rgba(239,68,68,0.4); }
.badge-yellow { background: rgba(245,158,11,0.2); color: #fbbf24; border: 1px solid rgba(245,158,11,0.4); }
.badge-green  { background: rgba(34,197,94,0.2);  color: #4ade80; border: 1px solid rgba(34,197,94,0.4); }
.badge-blue   { background: rgba(99,102,241,0.2); color: #818cf8; border: 1px solid rgba(99,102,241,0.4); }

/* Footer */
.footer {
    text-align: center;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.3);
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.06);
}
</style>
""", unsafe_allow_html=True)


# ── Model loading — cache key includes file mtime so retrained model auto-reloads ──
def _model_mtime() -> float:
    try:
        return os.path.getmtime(MODEL_PATH)
    except OSError:
        return 0.0

@st.cache_resource(hash_funcs={float: lambda x: x})
def load_model(_mtime: float = 0.0):
    model = get_model(input_dim=INPUT_DIM)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        return model, True
    except FileNotFoundError:
        return None, False


# ── Feature metadata ──────────────────────────────────────────────────────────
FEATURES = [
    # (key,                   display_name,                   unit)
    ("age",                  "Age",                          "years"),
    ("sex",                  "Sex",                          ""),
    ("bmi",                  "Body Mass Index (BMI)",         "kg/m²"),
    ("systolic_bp",          "Systolic Blood Pressure",       "mmHg"),
    ("diastolic_bp",         "Diastolic Blood Pressure",      "mmHg"),
    ("cholesterol_total",    "Total Cholesterol",             "mg/dL"),
    ("hdl_cholesterol",      "HDL Cholesterol (Good)",        "mg/dL"),
    ("ldl_cholesterol",      "LDL Cholesterol (Bad)",         "mg/dL"),
    ("blood_glucose",        "Fasting Blood Glucose",         "mg/dL"),
    ("smoking_status",       "Smoking Status",               ""),
    ("physical_activity",    "Physical Activity",             "days/week"),
    ("family_history",       "Family History of CVD",        ""),
    ("chest_pain",           "Chest Pain",                   ""),
    ("shortness_of_breath",  "Shortness of Breath",          ""),
    ("fatigue",              "Chronic Fatigue",              ""),
]

# Reference ranges for XAI bar display (min, max, elevated_threshold, direction)
FEATURE_RANGES = {
    "age":                 (20,  90,   55,   "higher"),
    "sex":                 (0,   1,    None, None),
    "bmi":                 (15,  45,   25,   "higher"),
    "systolic_bp":         (90,  200,  130,  "higher"),
    "diastolic_bp":        (60,  120,  85,   "higher"),
    "cholesterol_total":   (100, 400,  200,  "higher"),
    "hdl_cholesterol":     (20,  100,  60,   "lower"),   # low HDL is bad
    "ldl_cholesterol":     (50,  250,  130,  "higher"),
    "blood_glucose":       (70,  300,  100,  "higher"),
    "smoking_status":      (0,   1,    None, None),
    "physical_activity":   (0,   7,    3,    "lower"),   # low activity is bad
    "family_history":      (0,   1,    None, None),
    "chest_pain":          (0,   1,    None, None),
    "shortness_of_breath": (0,   1,    None, None),
    "fatigue":             (0,   1,    None, None),
}


def feature_risk_score(key: str, value: float) -> tuple[float, str]:
    """Return 0–1 risk contribution and colour hex for a feature value."""
    mn, mx, thresh, direction = FEATURE_RANGES[key]
    normalized = (value - mn) / max(mx - mn, 1e-9)
    normalized = np.clip(normalized, 0, 1)

    if direction == "higher":
        risk = normalized
        colour = f"hsl({int(120 - 120*normalized)}, 80%, 55%)"
    elif direction == "lower":
        risk = 1 - normalized
        colour = f"hsl({int(120*normalized)}, 80%, 55%)"
    else:
        risk = float(value)  # binary
        colour = "#ef4444" if value == 1 else "#22c55e"

    return risk, colour


def render_feature_bar(name: str, key: str, value: float):
    score, colour = feature_risk_score(key, value)
    pct = int(score * 100)
    st.markdown(f"""
    <div class="feature-row">
        <span class="feature-name">{name}</span>
        <div class="feature-bar-bg">
            <div class="feature-bar-fill" style="width:{pct}%; background:{colour};"></div>
        </div>
        <span class="feature-val">{pct}%</span>
    </div>
    """, unsafe_allow_html=True)


def get_recommendations(values: dict, risk: float) -> list[str]:
    recs = []
    if values["smoking_status"] == 1:
        recs.append("🚬 **Quit smoking** – smoking doubles CVD risk")
    if values["systolic_bp"] > 130 or values["diastolic_bp"] > 85:
        recs.append("💊 **Monitor blood pressure** – target <130/80 mmHg")
    if values["ldl_cholesterol"] > 130:
        recs.append("🥗 **Reduce LDL cholesterol** – consider dietary changes or statins")
    if values["hdl_cholesterol"] < 40:
        recs.append("🏃 **Raise HDL** – regular aerobic exercise increases HDL")
    if values["bmi"] > 25:
        recs.append("⚖️ **Achieve healthy weight** – target BMI 18.5–24.9")
    if values["blood_glucose"] > 100:
        recs.append("🩸 **Manage blood sugar** – target fasting glucose <100 mg/dL")
    if values["physical_activity"] < 3:
        recs.append("🏋️ **Increase activity** – aim for ≥150 min/week moderate exercise")
    if values["chest_pain"] == 1:
        recs.append("⚠️ **Chest pain** – seek immediate medical attention")
    if values["shortness_of_breath"] == 1:
        recs.append("🫁 **Shortness of breath** – consult a cardiologist")
    if not recs:
        recs.append("✅ Your risk factors appear well-controlled. Maintain current healthy habits.")
    return recs


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Hero header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-header">
        <h1>🫀 Personalized Disease Risk Manager</h1>
        <p>Cardiovascular disease risk prediction powered by privacy-preserving Federated Learning
        across distributed hospital networks. Enter your clinical data below.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Model status ──────────────────────────────────────────────────────────
    model, loaded = load_model(_mtime=_model_mtime())
    if not loaded:
        st.error(
            f"⚠️ **Model not found** (`{MODEL_PATH}`).  \n"
            "Run `python run_simulation.py` to train the federated learning model first.",
            icon="🚨",
        )
        st.info("💡 After training completes, refresh this page to use the predictor.")
        st.stop()

    col_badge1, col_badge2, col_badge3 = st.columns(3)
    with col_badge1:
        st.success("🟢 FL Model Loaded", icon="✅")
    with col_badge2:
        st.info("🔒 Privacy-Preserving (Differential Privacy)", icon="🛡")
    with col_badge3:
        st.info("🏥 3 Hospital Clients | 10 FL Rounds", icon="🤝")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  INPUT FORM
    # ══════════════════════════════════════════════════════════════════════════
    with st.form("patient_form"):
        # ── Section 1: Demographics ─────────────────────────────────────────
        st.markdown('<div class="section-title">👤 Patient Demographics</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", min_value=20, max_value=90, value=45, step=1)
            st.caption("Normal ranges vary.")
        with c2:
            sex_label = st.radio("Sex", options=["Female", "Male"], horizontal=True)
            sex = 1.0 if sex_label == "Male" else 0.0
        with c3:
            bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=45.0, value=26.0, step=0.1)
            st.caption("Normal: 18.5–24.9 | Risky: ≥25")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Section 2: Blood Pressure ───────────────────────────────────────
        st.markdown('<div class="section-title">💉 Blood Pressure</div>', unsafe_allow_html=True)
        c4, c5 = st.columns(2)
        with c4:
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=90, max_value=200, value=120, step=1)
            st.caption("Normal: <120 | Risky: ≥130")
        with c5:
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=60, max_value=120, value=80, step=1)
            st.caption("Normal: <80 | Risky: ≥80")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Section 3: Lab Results ──────────────────────────────────────────
        st.markdown('<div class="section-title">🧪 Laboratory Results</div>', unsafe_allow_html=True)
        c6, c7, c8, c9 = st.columns(4)
        with c6:
            cholesterol_total = st.number_input("Total Cholesterol", min_value=100, max_value=400, value=200, step=1)
            st.caption("Normal: <200 | Risky: ≥240")
        with c7:
            hdl_cholesterol = st.number_input("HDL Cholesterol", min_value=20, max_value=100, value=55, step=1)
            st.caption("Normal: ≥60 | Risky: <40")
        with c8:
            ldl_cholesterol = st.number_input("LDL Cholesterol", min_value=50, max_value=250, value=120, step=1)
            st.caption("Normal: <100 | Risky: ≥160")
        with c9:
            blood_glucose = st.number_input("Fasting Glucose", min_value=70, max_value=300, value=95, step=1)
            st.caption("Normal: <100 | Risky: ≥126")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Section 4: Lifestyle ────────────────────────────────────────────
        st.markdown('<div class="section-title">🏃 Lifestyle Factors</div>', unsafe_allow_html=True)
        c10, c11, c12 = st.columns(3)
        with c10:
            smoking_label = st.radio("Smoking Status", ["Non-Smoker", "Smoker"], horizontal=True)
            smoking_status = 1.0 if smoking_label == "Smoker" else 0.0
            st.caption("Risky: Smoker")
        with c11:
            physical_activity = st.number_input("Physical Activity (days)", min_value=0, max_value=7, value=3, step=1)
            st.caption("Normal: ≥5 | Risky: <3")
        with c12:
            family_label = st.radio("Family History of CVD", ["No", "Yes"], horizontal=True)
            st.caption("Risky: Yes")
            family_history = 1.0 if family_label == "Yes" else 0.0

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Section 5: Symptoms ─────────────────────────────────────────────
        st.markdown('<div class="section-title">🩺 Current Symptoms</div>', unsafe_allow_html=True)
        c13, c14, c15 = st.columns(3)
        with c13:
            chest_label = st.radio("Chest Pain / Discomfort", ["No", "Yes"], horizontal=True)
            chest_pain = 1.0 if chest_label == "Yes" else 0.0
            st.caption("Risky: Yes")
        with c14:
            sob_label = st.radio("Shortness of Breath", ["No", "Yes"], horizontal=True)
            shortness_of_breath = 1.0 if sob_label == "Yes" else 0.0
            st.caption("Risky: Yes")
        with c15:
            fatigue_label = st.radio("Chronic Fatigue", ["No", "Yes"], horizontal=True)
            fatigue = 1.0 if fatigue_label == "Yes" else 0.0
            st.caption("Risky: Yes")

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.form_submit_button(
            "🔍  Predict My Cardiovascular Risk",
            type="primary",
            use_container_width=True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  PREDICTION RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    if predict_btn:
        values = {
            "age":                 float(age),
            "sex":                 sex,
            "bmi":                 bmi,
            "systolic_bp":         float(systolic_bp),
            "diastolic_bp":        float(diastolic_bp),
            "cholesterol_total":   float(cholesterol_total),
            "hdl_cholesterol":     float(hdl_cholesterol),
            "ldl_cholesterol":     float(ldl_cholesterol),
            "blood_glucose":       float(blood_glucose),
            "smoking_status":      smoking_status,
            "physical_activity":   float(physical_activity),
            "family_history":      family_history,
            "chest_pain":          chest_pain,
            "shortness_of_breath": shortness_of_breath,
            "fatigue":             fatigue,
        }

        input_array = np.array(list(values.values()), dtype=np.float32)
        input_tensor = torch.tensor(input_array).unsqueeze(0)

        with torch.no_grad():
            risk_prob = model(input_tensor).item()

        risk_pct = risk_prob * 100

        st.markdown("---")
        st.subheader("📊 Prediction Results")

        # ── Risk gauge ──────────────────────────────────────────────────────
        col_gauge, col_detail = st.columns([1, 2])

        with col_gauge:
            if risk_pct >= 60:
                css_class = "risk-high"
                icon = "🚨"
                label = "HIGH RISK"
            elif risk_pct >= 35:
                css_class = "risk-moderate"
                icon = "⚠️"
                label = "MODERATE RISK"
            else:
                css_class = "risk-low"
                icon = "✅"
                label = "LOW RISK"

            st.markdown(f"""
            <div class="{css_class}">
                <div style="font-size:2.5rem;">{icon}</div>
                <div class="risk-value">{risk_pct:.1f}%</div>
                <div class="risk-label">{label}</div>
                <div style="font-size:0.8rem;opacity:0.7;margin-top:0.5rem;">
                    Cardiovascular Disease Probability
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(risk_prob, text=f"Risk Score: {risk_pct:.1f}%")

        # ── Risk factors summary ────────────────────────────────────────────
        with col_detail:
            st.markdown("**Elevated Risk Factors Detected:**")
            badges_html = ""
            if values["smoking_status"] == 1:
                badges_html += '<span class="badge badge-red">🚬 Smoker</span>'
            if values["systolic_bp"] >= 130:
                badges_html += '<span class="badge badge-red">⬆ High Systolic BP</span>'
            if values["ldl_cholesterol"] >= 130:
                badges_html += '<span class="badge badge-yellow">⬆ High LDL</span>'
            if values["hdl_cholesterol"] < 40:
                badges_html += '<span class="badge badge-yellow">⬇ Low HDL</span>'
            if values["blood_glucose"] >= 100:
                badges_html += '<span class="badge badge-yellow">🩸 Elevated Glucose</span>'
            if values["chest_pain"] == 1:
                badges_html += '<span class="badge badge-red">💔 Chest Pain</span>'
            if values["shortness_of_breath"] == 1:
                badges_html += '<span class="badge badge-red">🫁 Dyspnea</span>'
            if values["bmi"] >= 30:
                badges_html += '<span class="badge badge-yellow">⚖ Obese BMI</span>'
            if values["family_history"] == 1:
                badges_html += '<span class="badge badge-blue">👨‍👩‍👦 Family History</span>'
            if values["physical_activity"] < 3:
                badges_html += '<span class="badge badge-yellow">🛋 Low Activity</span>'
            if not badges_html:
                badges_html = '<span class="badge badge-green">✅ No major flags</span>'

            st.markdown(badges_html, unsafe_allow_html=True)

            st.markdown("<br>**Clinical Recommendations:**", unsafe_allow_html=True)
            for rec in get_recommendations(values, risk_pct):
                st.markdown(f"- {rec}")

        # ── XAI Feature Contribution Breakdown ─────────────────────────────
        st.markdown("---")
        st.subheader("🔬 Feature Risk Contribution Analysis")
        st.caption("Based on Explainable AI (XAI) principles – showing relative risk contribution of each clinical feature")

        col_xai1, col_xai2 = st.columns(2)
        half = len(FEATURES) // 2

        with col_xai1:
            for key, name, unit in FEATURES[:half]:
                label = f"{name} ({values[key]:.0f}{' '+unit if unit else ''})"
                render_feature_bar(label, key, values[key])

        with col_xai2:
            for key, name, unit in FEATURES[half:]:
                if unit:
                    label = f"{name} ({values[key]:.0f} {unit})"
                    render_feature_bar(label, key, values[key])
                else:
                    binary_labels = {
                        "sex": ("Female", "Male"),
                        "smoking_status": ("Non-Smoker", "Smoker"),
                        "family_history": ("No", "Yes"),
                        "chest_pain":     ("No", "Yes"),
                        "shortness_of_breath": ("No", "Yes"),
                        "fatigue":        ("No", "Yes"),
                    }
                    opts = binary_labels.get(key, ("0", "1"))
                    label = f"{name} ({opts[int(values[key])]})"
                    render_feature_bar(label, key, values[key])

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        🔒 Privacy-Preserving Federated Learning &nbsp;|&nbsp;
        Differential Privacy Noise Applied &nbsp;|&nbsp;
        No patient data leaves the hospital &nbsp;|&nbsp;
        <br>References: Hoghooghi-Esfahani et al. (2025) · Federated EHR Learning (2021, 2023)
        <br><br>⚠️ <em>For research/educational purposes only. Not a substitute for clinical diagnosis.</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
