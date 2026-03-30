import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ── Real clinical features for Cardiovascular Disease Risk Prediction ──────────
# Based on:
#   • Hoghooghi-Esfahani et al. (2025) – XAI in disease prediction
#   • Federated learning for privacy-preserving EHR (2021, 2023)
FEATURE_NAMES = [
    "age",                  # Years (20–90)
    "sex",                  # 0 = Female, 1 = Male
    "bmi",                  # Body Mass Index (15–45)
    "systolic_bp",          # Systolic Blood Pressure mmHg (90–200)
    "diastolic_bp",         # Diastolic Blood Pressure mmHg (60–120)
    "cholesterol_total",    # Total Cholesterol mg/dL (100–400)
    "hdl_cholesterol",      # HDL (good) Cholesterol mg/dL (20–100)
    "ldl_cholesterol",      # LDL (bad) Cholesterol mg/dL (50–250)
    "blood_glucose",        # Fasting Blood Glucose mg/dL (70–300)
    "smoking_status",       # 0 = Non-smoker, 1 = Smoker
    "physical_activity",    # Days/week of moderate exercise (0–7)
    "family_history",       # Family history of CVD: 0 = No, 1 = Yes
    "chest_pain",           # 0 = No, 1 = Yes
    "shortness_of_breath",  # 0 = No, 1 = Yes
    "fatigue",              # Chronic fatigue: 0 = No, 1 = Yes
]

INPUT_DIM = len(FEATURE_NAMES)  # 15


def generate_ehr_data(num_samples: int = 10_000, random_state: int = 42) -> pd.DataFrame:
    """Generate clinically realistic EHR data for CVD risk prediction."""
    rng = np.random.default_rng(random_state)

    # ── Demographics ──────────────────────────────────────────────────────────
    age = rng.integers(20, 91, size=num_samples).astype(float)
    sex = rng.integers(0, 2, size=num_samples).astype(float)         # 0=F, 1=M

    # ── Anthropometrics ───────────────────────────────────────────────────────
    bmi = rng.normal(27.5, 5.5, size=num_samples).clip(15, 45)

    # ── Blood Pressure (correlated with age & BMI) ───────────────────────────
    sbp_base = 100 + 0.5 * age + 0.8 * (bmi - 25)
    systolic_bp  = (sbp_base  + rng.normal(0, 12, size=num_samples)).clip(90, 200)
    diastolic_bp = (systolic_bp * 0.6 + rng.normal(5, 8, size=num_samples)).clip(60, 120)

    # ── Lipids (mg/dL) ────────────────────────────────────────────────────────
    cholesterol_total = rng.normal(200, 40, size=num_samples).clip(100, 400)
    hdl_cholesterol   = rng.normal(55,  15, size=num_samples).clip(20,  100)
    ldl_cholesterol   = (cholesterol_total - hdl_cholesterol - rng.normal(30, 8, size=num_samples)).clip(50, 250)

    # ── Blood Glucose (mg/dL) ─────────────────────────────────────────────────
    blood_glucose = rng.normal(100, 30, size=num_samples).clip(70, 300)

    # ── Lifestyle & Symptoms ──────────────────────────────────────────────────
    smoking_status   = rng.binomial(1, 0.25,  size=num_samples).astype(float)
    physical_activity = rng.integers(0, 8,   size=num_samples).astype(float)
    family_history   = rng.binomial(1, 0.30,  size=num_samples).astype(float)
    chest_pain       = rng.binomial(1, 0.15,  size=num_samples).astype(float)
    short_breath     = rng.binomial(1, 0.18,  size=num_samples).astype(float)
    fatigue          = rng.binomial(1, 0.22,  size=num_samples).astype(float)

    # ── Target: Cardiovascular Disease Risk (clinically grounded logistic) ────
    # Coefficients inspired by Framingham Risk Score & reviewed literature
    # Intercept tuned to produce ~35-40% CVD-positive rate for realistic ML training
    log_odds = (
        -4.5
        + 0.06  * age
        + 0.20  * sex
        + 0.08  * (bmi - 25).clip(0, None)
        + 0.025 * (systolic_bp - 120).clip(0, None)
        + 0.015 * (diastolic_bp - 80).clip(0, None)
        + 0.010 * (cholesterol_total - 200).clip(0, None)
        - 0.040 * (hdl_cholesterol - 60).clip(0, None)   # low HDL raises risk
        + 0.015 * (ldl_cholesterol - 130).clip(0, None)
        + 0.008 * (blood_glucose - 100).clip(0, None)
        + 0.70  * smoking_status
        - 0.18  * physical_activity
        + 0.50  * family_history
        + 0.90  * chest_pain
        + 0.65  * short_breath
        + 0.40  * fatigue
        + rng.normal(0, 0.4, size=num_samples)   # biological noise
    )
    prob = 1 / (1 + np.exp(-log_odds))
    target = (prob >= 0.5).astype(int)

    df = pd.DataFrame({
        "age":                 age,
        "sex":                 sex,
        "bmi":                 bmi.round(1),
        "systolic_bp":         systolic_bp.round(0),
        "diastolic_bp":        diastolic_bp.round(0),
        "cholesterol_total":   cholesterol_total.round(1),
        "hdl_cholesterol":     hdl_cholesterol.round(1),
        "ldl_cholesterol":     ldl_cholesterol.round(1),
        "blood_glucose":       blood_glucose.round(1),
        "smoking_status":      smoking_status,
        "physical_activity":   physical_activity,
        "family_history":      family_history,
        "chest_pain":          chest_pain,
        "shortness_of_breath": short_breath,
        "fatigue":             fatigue,
        "target":              target,
    })
    return df


def generate_synthetic_ehr_data(num_samples: int = 10_000, num_clients: int = 3, random_state: int = 42):
    """Generate realistic EHR data and partition among FL clients."""
    print(f"Generating {num_samples} clinically realistic EHR patient records...")
    df = generate_ehr_data(num_samples=num_samples, random_state=random_state)

    # 80/20 train-test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state)

    os.makedirs("data", exist_ok=True)
    test_df.to_csv("data/server_test_data.csv", index=False)
    print(f"  Saved server test set  → data/server_test_data.csv  ({len(test_df):,} records)")

    # Split training data across FL clients (simulate hospital data silos)
    chunks = np.array_split(train_df, num_clients)
    for i, chunk in enumerate(chunks):
        path = f"data/client_{i+1}_data.csv"
        chunk.to_csv(path, index=False)
        pos_rate = chunk["target"].mean() * 100
        print(f"  Saved client {i+1} data    → {path}  ({len(chunk):,} records, {pos_rate:.1f}% CVD positive)")

    print(f"\nFeatures ({INPUT_DIM} total): {', '.join(FEATURE_NAMES)}")
    print(f"Class balance (train): {train_df['target'].mean()*100:.1f}% CVD positive")


if __name__ == "__main__":
    generate_synthetic_ehr_data()
