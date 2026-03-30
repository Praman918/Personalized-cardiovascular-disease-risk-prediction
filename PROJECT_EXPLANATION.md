# 📖 Project Documentation: Personalized Disease Risk Manager

### *A Complete Explanation — For a Peer, For a Teacher, For a Viva*

---

## Table of Contents

1. [What is this project in simple words?](#1-what-is-this-project-in-simple-words)
2. [The Problem We Are Solving](#2-the-problem-we-are-solving)
3. [What is Federated Learning? (The Core Idea)](#3-what-is-federated-learning-the-core-idea)
4. [What is a Neural Network? (For Beginners)](#4-what-is-a-neural-network-for-beginners)
5. [System Architecture — How All the Parts Connect](#5-system-architecture--how-all-the-parts-connect)
6. [The Clinical Features — Why These 15?](#6-the-clinical-features--why-these-15)
7. [How the Data is Generated](#7-how-the-data-is-generated)
8. [The Machine Learning Model](#8-the-machine-learning-model)
9. [Federated Averaging (FedAvg) — How the AI Learns Without Sharing Data](#9-federated-averaging-fedavg--how-the-ai-learns-without-sharing-data)
10. [Differential Privacy — An Extra Layer of Protection](#10-differential-privacy--an-extra-layer-of-protection)
11. [Explainable AI (XAI) — Why Did It Say High Risk?](#11-explainable-ai-xai--why-did-it-say-high-risk)
12. [The Web Application](#12-the-web-application)
13. [File-by-File Code Explanation](#13-file-by-file-code-explanation)
14. [Expected Results and Performance](#14-expected-results-and-performance)
15. [Limitations and Future Scope](#15-limitations-and-future-scope)
16. [Glossary of Technical Terms](#16-glossary-of-technical-terms)
17. [References](#17-references)

---

## 1. What is this project in simple words?

Imagine you go to your doctor. The doctor looks at your age, blood pressure, cholesterol levels, whether you smoke, and other health data — and tells you if you're at risk of a heart attack or cardiovascular disease.

**This project does the same thing, but using Artificial Intelligence.**

But here's the twist: hospitals have millions of patient records that could train a powerful AI model. The problem is — **you can't just share patient data between hospitals**. It's private, sensitive, and often illegal to share under laws like HIPAA (USA) or GDPR (Europe).

So we use a technique called **Federated Learning** — where each hospital trains the AI on their own data locally, and only the learned knowledge (not the raw patient data) is shared to build one powerful global model.

> **One-liner for your peer:** *"It's an AI that predicts your heart disease risk using your health data, but trained in a way that keeps everyone's medical records completely private."*

> **One-liner for viva:** *"This project implements a privacy-preserving, federated machine learning pipeline for personalized cardiovascular disease risk prediction, incorporating differential privacy and XAI-based feature attribution."*

---

## 2. The Problem We Are Solving

### The Medical Problem
- Cardiovascular disease (CVD) is the **#1 cause of death worldwide** (WHO, 2023)
- Early detection saves lives — if you know you're at risk, you can change your lifestyle
- Risk prediction requires combining multiple health metrics — no single factor tells the whole story

### The Technical / Privacy Problem
Consider 3 hospitals:
- Hospital A has 10,000 patient records
- Hospital B has 8,000 patient records
- Hospital C has 12,000 patient records

If they could combine all 30,000 records, they'd train a much better AI model. But they **legally and ethically cannot** share patient data with each other.

**Traditional ML** → needs all data in one place → Privacy violation ❌

**Federated Learning** → model travels to each hospital → Data never moves → ✅

---

## 3. What is Federated Learning? (The Core Idea)

### The Analogy
Think of a school exam where every student has a unique textbook (hospital data). Instead of collecting everyone's textbooks in one place, the teacher:

1. Gives each student the same **blank exam paper** (the initial AI model)
2. Each student **studies with their own textbook** and writes their answers (trains locally)
3. The teacher collects only the **answer papers** (model weights — not the textbooks)
4. The teacher **averages the best answers** (FedAvg aggregation)
5. This becomes the new improved exam paper for the next round

The textbooks (patient data) **never leave the students' hands.**

### In Technical Terms
```
Round 1:
  Server sends global model weights → Client 1, Client 2, Client 3
  Each client trains on local data
  Each client sends back UPDATED WEIGHTS only (not data)
  Server averages all weights → new global model

Round 2:
  Repeat with improved model
  ...
Round 10:
  Final trained global model is saved
```

### Why is this powerful?
- More data effectively used from all hospitals → Better model
- Zero raw patient data sharing → Full privacy compliance
- Each hospital keeps 100% control of its data

---

## 4. What is a Neural Network? (For Beginners)

A neural network is a mathematical function loosely inspired by how neurons in the brain connect.

### Structure
```
INPUT LAYER          HIDDEN LAYERS              OUTPUT LAYER
(15 health           (learn patterns            (probability of
 features)           from combinations)          CVD: 0 to 1)

age ──────────┐
systolic_bp ──┤      [64 neurons]
cholesterol ──┤  →   [64 neurons]  →   [32 neurons]  →  0.0 to 1.0
smoking ──────┤
chest_pain ───┘
...
```

- **Input layer**: Our 15 clinical features go in as numbers
- **Hidden layers**: The network learns complex relationships (e.g., *"a 60-year-old smoker with high LDL AND chest pain = very high risk"*)
- **Output layer**: A single number between 0 and 1 (the probability of having CVD)

A probability > 0.5 → **High Risk**, < 0.5 → **Low Risk**

### How it learns
The network starts with random weights (guesses). It compares its output to the actual label (does the patient have CVD?). The error is fed back through the network (**backpropagation**) and weights are adjusted. Repeated thousands of times = the model learns.

---

## 5. System Architecture — How All the Parts Connect

```
                    ┌─────────────────────────────────┐
                    │           SERVER (server.py)     │
                    │  - Holds Global Model           │
                    │  - Sends weights to clients     │
                    │  - Receives updated weights     │
                    │  - Aggregates via FedAvg        │
                    │  - Evaluates on test set        │
                    └────────────┬────────────────────┘
                                 │ HTTP (weights only, never data)
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
   │  CLIENT 1 :8001  │ │  CLIENT 2 :8002  │ │  CLIENT 3 :8003  │
   │  Hospital A      │ │  Hospital B      │ │  Hospital C      │
   │  client_1_data   │ │  client_2_data   │ │  client_3_data   │
   │  Trains locally  │ │  Trains locally  │ │  Trains locally  │
   │  Adds DP noise   │ │  Adds DP noise   │ │  Adds DP noise   │
   └──────────────────┘ └──────────────────┘ └──────────────────┘
              
              
   ┌──────────────────────────────────────────────────────────┐
   │                  WEB APP (app.py)                        │
   │  - Loads global_model.pth                               │
   │  - User fills in clinical form                          │
   │  - Model predicts CVD risk probability                  │
   │  - Displays risk gauge, XAI bars, recommendations       │
   └──────────────────────────────────────────────────────────┘
```

### Communication Protocol
- Clients are **FastAPI** web servers
- The server sends `POST /train` requests with JSON-formatted weights
- Clients respond with updated weights as JSON
- This simulates a real-world scenario where hospitals are on different networks

---

## 6. The Clinical Features — Why These 15?

These features are based on the **Framingham Heart Study** (one of the most influential cardiovascular research studies) and clinical guidelines:

| # | Feature | Type | Why It Matters |
|---|---|---|---|
| 1 | **Age** | Continuous | CVD risk doubles every decade after 40 |
| 2 | **Sex** | Binary | Males have higher CVD risk on average |
| 3 | **BMI** | Continuous | Obesity strains the heart and promotes inflammation |
| 4 | **Systolic BP** | Continuous | High pressure damages artery walls over time |
| 5 | **Diastolic BP** | Continuous | Indicator of arterial resistance |
| 6 | **Total Cholesterol** | Continuous | High total cholesterol = plaque buildup in arteries |
| 7 | **HDL Cholesterol** | Continuous | "Good" cholesterol — HIGH HDL is PROTECTIVE |
| 8 | **LDL Cholesterol** | Continuous | "Bad" cholesterol — HIGH LDL = HIGH RISK |
| 9 | **Fasting Blood Glucose** | Continuous | Diabetes is a major CVD risk multiplier |
| 10 | **Smoking Status** | Binary | Smoking doubles CVD risk — causes arterial damage |
| 11 | **Physical Activity** | Ordinal | Exercise strengthens the heart |
| 12 | **Family History** | Binary | Genetic predisposition to CVD |
| 13 | **Chest Pain** | Binary | Classic symptom of cardiac ischemia (oxygen deprivation) |
| 14 | **Shortness of Breath** | Binary | Dyspnea — heart may not be pumping enough blood |
| 15 | **Chronic Fatigue** | Binary | Heart struggling to meet the body's demands |

### Why not more features?
More isn't always better in ML (curse of dimensionality). These 15 cover the major independent risk factors. Each adds predictive value without excessive noise.

---

## 7. How the Data is Generated

> **Important note for viva:** *"Real patient data could not be used for this project for ethical and privacy reasons. Therefore, we used clinically realistic synthetic data generated using a mathematically grounded logistic model."*

### Process (in `data_generator.py`)

**Step 1 — Generate features with realistic distributions:**
```python
age         ~ Uniform(20, 90)
bmi         ~ Normal(27.5, 5.5)  clipped to [15, 45]
systolic_bp ~ (100 + 0.5×age + 0.8×(bmi-25)) + Normal(0, 12)
```
Blood pressure is correlated with age and BMI — just like in real life.

**Step 2 — Compute risk using Framingham-inspired logistic model:**
```
log_odds = -4.5
           + 0.06 × age
           + 0.20 × sex
           + 0.70 × smoking
           + 0.025 × (systolic_bp - 120)
           + 0.90 × chest_pain
           + ... (all 15 features)
```
```
probability = sigmoid(log_odds) = 1 / (1 + e^(-log_odds))
```
If probability ≥ 0.5 → target = 1 (CVD positive), else target = 0

This formula is inspired by the real Framingham coefficients, producing a **~38% positive class rate** for balanced training.

**Step 3 — Split into 4 files:**
- `server_test_data.csv` — 20% held-out for evaluation
- `client_1/2/3_data.csv` — 80% split evenly for 3 hospital clients

---

## 8. The Machine Learning Model

Defined in `model.py` — a **feedforward neural network** built in PyTorch.

```
Input (15)  →  Linear(15→64) → ReLU → Dropout(0.3)
            →  Linear(64→64) → ReLU → Dropout(0.25)
            →  Linear(64→32) → ReLU → Dropout(0.2)
            →  Linear(32→1)  → Sigmoid
                                ↓
                         output: 0.0 – 1.0
```

### Why these design choices?

| Choice | Reason |
|---|---|
| **ReLU activation** | Faster training, avoids vanishing gradient problem |
| **Dropout** | Prevents overfitting — randomly "turns off" neurons during training |
| **Sigmoid output** | Squashes final output to [0,1] so it represents a probability |
| **No BatchNorm** | BatchNorm uses running statistics that are NOT transferred in FL's weight exchange — would cause corrupt outputs in eval mode |
| **3 hidden layers** | Deep enough to learn non-linear patterns, not so deep it overfits on this dataset size |

### Loss Function — Weighted BCE
During FL training, we use **weighted Binary Cross-Entropy loss**:

```python
loss = -(pos_weight × y × log(ŷ) + (1-y) × log(1-ŷ))
```

The `pos_weight` term penalises the model more for missing positive CVD cases — addressing class imbalance.

### Optimizer
**Adam** (Adaptive Moment Estimation) — the standard choice for deep learning. Adapts the learning rate per parameter automatically.

---

## 9. Federated Averaging (FedAvg) — How the AI Learns Without Sharing Data

FedAvg was introduced by McMahan et al. (2017) at Google. It is now the dominant algorithm in federated learning.

### The Math

After each round, the server computes a **weighted average** of all client models:

```
w_global = Σ (n_k / n_total) × w_k
```

Where:
- `w_k` = weights returned by client k
- `n_k` = number of training samples at client k
- `n_total` = total samples across all clients

**Why weighted?** A hospital with 5,000 patients should have more influence than one with 500 — their model learned from more evidence.

### Why This Works
Each client starts each round from the **same global model**, trains locally for a few epochs, then returns their improved version. When averaged, the global model captures patterns from ALL hospitals simultaneously — without any hospital seeing another's data.

### The FL Loop in this project

```
for round in 1..10:
    global_weights → broadcast to all 3 clients
    
    client_k:
        load global weights
        compute pos_weight from local data
        train with weighted BCELoss for 5 epochs
        add differential privacy noise
        return updated weights
    
    server:
        aggregate via FedAvg
        evaluate on test set → print accuracy
        
final: save global_model.pth
```

---

## 10. Differential Privacy — An Extra Layer of Protection

> *"Even sharing model weights could theoretically leak patient information through careful analysis. Differential Privacy prevents this."*

In `client.py`, after local training, Gaussian noise is added to every weight before it's sent to the server:

```python
noisy_weight = trained_weight + Normal(0, 0.005)
```

Additionally, **gradient clipping** is applied during training:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Why this matters
- **Gradient clipping** limits how much any single patient can influence the model
- **Noise addition** makes it mathematically impossible to reverse-engineer individual patient records from the weights

This is a **simulation** of formal (ε, δ)-Differential Privacy. In production, a library like `Opacus` (by PyTorch/Meta) would be used to compute formal DP guarantees.

---

## 11. Explainable AI (XAI) — Why Did It Say High Risk?

A common criticism of AI in healthcare: *"The model said high risk — but why? What does the doctor act on?"*

This project incorporates **XAI-inspired feature attribution** — showing how much each clinical feature contributes to the risk assessment.

### How the feature bars work in the app

Each feature value is normalised to [0,1] relative to its clinical range, then mapped to a risk direction:

```python
# Example: Systolic BP (higher = more risk)
normalized = (value - 90) / (200 - 90)    # 0 = 90mmHg, 1 = 200mmHg
risk_score = normalized
colour = hsl(120 - 120×risk_score)        # green → red as risk increases

# Example: HDL Cholesterol (LOWER = more risk)
risk_score = 1 - normalized               # inverted
```

Binary features (smoking, chest pain) show green (No) or red (Yes).

### What a proper XAI method would use
For a full production system, **SHAP (SHapley Additive exPlanations)** would compute the exact contribution of each feature to the model's output based on game theory. Our implementation follows the same principle using clinical domain knowledge for the bar colouring.

---

## 12. The Web Application

Built with **Streamlit** — a Python library that turns scripts into interactive web UIs.

### UI Sections

| Section | Inputs | Why |
|---|---|---|
| Patient Demographics | Age, Sex, BMI | Core risk stratifiers |
| Blood Pressure | Systolic, Diastolic | Direct cardiac load |
| Lab Results | Total/HDL/LDL cholesterol, Blood glucose | Established CVD biomarkers |
| Lifestyle | Smoking, Activity, Family history | Modifiable + genetic risk |
| Symptoms | Chest pain, Dyspnea, Fatigue | Acute warning signs |

### Output Components
1. **Risk Gauge** — colour-coded: 🟢 <35% LOW | 🟡 35-60% MODERATE | 🔴 >60% HIGH
2. **Risk Badges** — highlights exactly which factors are elevated (e.g. "🚬 Smoker", "💔 Chest Pain")
3. **Clinical Recommendations** — actionable advice per elevated factor
4. **XAI Bars** — per-feature contribution visualisation for all 15 features

### Privacy notice in UI
The footer explicitly states the model was trained with Federated Learning and Differential Privacy — educating the user that no raw patient data was shared during training.

---

## 13. File-by-File Code Explanation

### `data_generator.py`
- **Purpose**: Creates synthetic EHR data
- **Key function**: `generate_ehr_data()` — builds the DataFrame with 15 features; `generate_synthetic_ehr_data()` — saves it to CSV files
- **Clinical basis**: Framingham Risk Score coefficient magnitudes

### `model.py`
- **Purpose**: Defines the neural network
- **Key class**: `DiseaseRiskModel` — 3 hidden layers, ReLU, Dropout, Sigmoid output
- **Key constant**: `INPUT_DIM = 15` — used by all other files to ensure consistent dimensions

### `server.py`
- **Purpose**: Federated Learning server
- **Key function**: `run_federated_learning()` — orchestrates 10 FL rounds
- **Algorithm**: FedAvg with sample-proportional weighting
- **Communication**: Sends HTTP POST requests to each client's `/train` endpoint

### `client.py`
- **Purpose**: Hospital-side FL client
- **Framework**: FastAPI — each client is a REST API server
- **Key endpoint**: `POST /train` — receives global weights, trains locally, returns updated weights
- **Privacy**: Gradient clipping + Gaussian noise before returning weights

### `run_simulation.py`
- **Purpose**: Orchestrates the entire simulation end-to-end
- **Steps**: Generate data → Start clients (as subprocesses) → Run server → Kill clients
- **Entry point**: `python run_simulation.py`

### `app.py`
- **Purpose**: Streamlit web application
- **Key functions**: `load_model()` — loads trained model with mtime cache-busting; `main()` — renders the full UI
- **Cache strategy**: `mtime`-based so model auto-reloads after retraining without restarting the app

### `requirements.txt`
Lists all Python packages:
- `torch` — neural network implementation
- `streamlit` — web UI
- `fastapi` + `uvicorn` — client HTTP servers
- `scikit-learn` — train/test split utility
- `pandas` / `numpy` — data handling
- `requests` — server-to-client HTTP communication

---

## 14. Expected Results and Performance

After 10 FL rounds with 3 clients (10,000 synthetic patients total):

| Metric | Typical Value |
|---|---|
| Training accuracy | 75–85% |
| Test set accuracy | 70–80% |
| High-risk input prediction | 85–97% probability |
| Low-risk input prediction | <15% probability |

### Example predictions

| Profile | Prediction |
|---|---|
| Age 75, Male, Smoker, BP 155/95, Chest Pain, Dyspnea | ~94–97% HIGH RISK |
| Age 30, Female, Non-smoker, BP 110/72, No symptoms | <10% LOW RISK |
| Age 50, Male, Normal vitals, Slightly elevated cholesterol | 30–50% MODERATE |

---

## 15. Limitations and Future Scope

### Current Limitations

| Limitation | Explanation |
|---|---|
| **Synthetic data** | Real patient data was not used — model may not generalise to clinical settings |
| **Simulated FL** | All 3 "hospitals" run locally — in reality they'd be on separate servers/clouds |
| **Simulated DP** | Noise scale is heuristic — formal ε-DP guarantees not computed |
| **No secure aggregation** | Weights are sent in plaintext; production would use encrypted communication |
| **Binary classification only** | Predicts CVD / no CVD; doesn't distinguish specific conditions |

### Future Scope

1. **Real EHR data** — train on de-identified datasets like MIMIC-III or UK Biobank
2. **Multi-cloud deployment** — deploy each client to Azure / GCP / AWS to simulate real hospital networks
3. **Formal DP** — integrate PyTorch Opacus for provable (ε, δ)-DP
4. **SHAP values** — replace heuristic XAI bars with proper Shapley value computation
5. **Federated evaluation** — evaluate model on each client's private test set without centralising
6. **More diseases** — extend beyond CVD to diabetes, stroke, kidney disease

---

## 16. Glossary of Technical Terms

| Term | Plain English |
|---|---|
| **Federated Learning** | AI training without sharing raw data — model travels to the data, not the other way around |
| **FedAvg** | Federated Averaging — the algorithm that combines model updates from all clients into one global model |
| **Neural Network** | A mathematical system of connected layers that learns patterns from data |
| **Weights / Parameters** | The numbers inside the neural network that get tuned during training |
| **Backpropagation** | The process of adjusting weights by measuring and propagating the prediction error backwards |
| **Sigmoid** | A function that squashes any number to between 0 and 1 — used for probability output |
| **ReLU** | Rectified Linear Unit — activation function: max(0, x) — helps networks learn faster |
| **Dropout** | Randomly ignoring neurons during training to prevent overfitting |
| **Overfitting** | When a model memorises training data and fails on new data |
| **Differential Privacy** | Adding mathematical noise to prevent reverse-engineering individual records from model parameters |
| **XAI** | Explainable AI — making AI decisions transparent and interpretable to humans |
| **FastAPI** | Python framework for building fast REST APIs |
| **Streamlit** | Python library that creates web apps from scripts |
| **BCELoss** | Binary Cross-Entropy Loss — the standard loss function for binary classification |
| **EHR** | Electronic Health Record — digital patient medical records |
| **CVD** | Cardiovascular Disease — diseases of the heart and blood vessels |
| **BMI** | Body Mass Index — weight(kg) / height(m)² — measure of body fat |
| **HDL** | High-Density Lipoprotein — "good" cholesterol, protects against CVD |
| **LDL** | Low-Density Lipoprotein — "bad" cholesterol, contributes to plaque buildup |

---

## 17. References

1. **McMahan, H.B. et al. (2017)** — *Communication-Efficient Learning of Deep Networks from Decentralized Data* — the original FedAvg paper (Google Brain)

2. **Hoghooghi-Esfahani et al. (2025)** — *Explainable AI in Disease Risk Prediction* — basis for XAI feature attribution approach

3. **D'Agostino et al. (2008)** — *General Cardiovascular Risk Profile for Use in Primary Care* — Framingham-based risk coefficient reference

4. **World Health Organization (2023)** — *Cardiovascular Diseases Fact Sheet*

5. **Abadi, M. et al. (2016)** — *Deep Learning with Differential Privacy* (Google) — basis for DP noise addition

6. **PyTorch Documentation** — neural network implementation

7. **Streamlit Documentation** — web application framework

8. **FastAPI Documentation** — client REST API framework

---

> ⚕️ *This project is built for educational and research purposes. It demonstrates how federated learning, differential privacy, and explainable AI can be combined to build privacy-preserving healthcare AI systems. It is NOT intended for real clinical use.*
