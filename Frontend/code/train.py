import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
 
warnings.filterwarnings("ignore")
 
# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR   = os.path.join(FRONTEND_DIR, "models")
DATA_DIR     = os.path.join(FRONTEND_DIR, "data")
PIMA_CSV     = os.path.join(SCRIPT_DIR, "..", "..", "code", "PIMA", "pima_diabetes.csv")
LUNG_CSV     = os.path.join(DATA_DIR, "lung_cancer.csv")
 
os.makedirs(MODELS_DIR, exist_ok=True)
 
print(f"Frontend dir : {FRONTEND_DIR}")
print(f"Models dir   : {MODELS_DIR}")
print()
 
np.random.seed(42)
N = 1200   # synthetic samples per model
 
 
def save_model(m, name, X_te, y_te, filename):
    acc = accuracy_score(y_te, m.predict(X_te))
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(m, path)
    print(f"   ✅ Saved → {filename}  (accuracy: {acc*100:.1f}%)")
 
 
def train_rf(X, y, stratify=False):
    strat = y if stratify else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat)
    m = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
    m.fit(X_tr, y_tr)
    return m, X_te, y_te
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 1. DIABETES  — uses real PIMA CSV from the project
#    app.py input: [[Pregnancies, Glucose, BloodPressure, SkinThickness,
#                    Insulin, BMI, DiabetesPedigreeFunction, Age]]
# ══════════════════════════════════════════════════════════════════════════════
print("1️⃣  Training Diabetes model …")
try:
    # Search multiple possible locations
    pima_candidates = [
        PIMA_CSV,
        os.path.join(DATA_DIR, "pima_diabetes.csv"),
        os.path.join(FRONTEND_DIR, "..", "code", "PIMA", "pima_diabetes.csv"),
        os.path.join(SCRIPT_DIR, "..", "PIMA", "pima_diabetes.csv"),
    ]
    pima_found = next((p for p in pima_candidates if os.path.exists(p)), None)
    if pima_found is None:
        raise FileNotFoundError(
            "PIMA CSV not found. Copy pima_diabetes.csv into Frontend/data/ and retry.")
    df = pd.read_csv(pima_found)
    print(f"   Using PIMA CSV: {pima_found}")
 
    DIABETES_COLS = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                     "Insulin","BMI","DiabetesPedigreeFunction","Age"]
    X = df[DIABETES_COLS]
    y = df["Outcome"]
    m, X_te, y_te = train_rf(X, y, stratify=True)
    save_model(m, "Diabetes", X_te, y_te, "diabetes_model.sav")
except Exception as e:
    print(f"   ❌ Diabetes failed: {e}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 2. HEART DISEASE  — synthetic, UCI Cleveland distribution
#    app.py input (plain list, 13 values, no column names):
#    [age, sex, cp, trestbps, chol, fbs, restecg,
#     thalach, exang, oldpeak, slope, ca, thal]
# ══════════════════════════════════════════════════════════════════════════════
print("\n2️⃣  Training Heart Disease model …")
try:
    age      = np.random.normal(54, 9, N).clip(29, 77).astype(int)
    sex      = np.random.binomial(1, 0.68, N)
    cp       = np.random.choice([0,1,2,3], N, p=[0.47,0.17,0.28,0.08])
    trestbps = np.random.normal(131, 17, N).clip(94, 200).astype(int)
    chol     = np.random.normal(246, 51, N).clip(126, 564).astype(int)
    fbs      = np.random.binomial(1, 0.15, N)
    restecg  = np.random.choice([0,1,2], N, p=[0.50,0.01,0.49])
    thalach  = np.random.normal(149, 22, N).clip(71, 202).astype(int)
    exang    = np.random.binomial(1, 0.33, N)
    oldpeak  = np.abs(np.random.normal(1.0, 1.2, N)).clip(0, 6.2)
    slope    = np.random.choice([0,1,2], N, p=[0.21,0.46,0.33])
    ca       = np.random.choice([0,1,2,3], N, p=[0.59,0.23,0.12,0.06])
    thal     = np.random.choice([0,1,2], N, p=[0.55,0.06,0.39])
 
    risk   = ((age > 55)*0.30 + exang*0.30 + (ca > 0)*0.20
              + (oldpeak > 2)*0.20 + np.random.normal(0, 0.08, N))
    target = (risk > 0.45).astype(int)
 
    # Train as plain numpy array (app.py sends [[v1,v2,...,v13]])
    X = np.column_stack([age, sex, cp, trestbps, chol, fbs, restecg,
                         thalach, exang, oldpeak, slope, ca, thal])
    y = target
    m, X_te, y_te = train_rf(X, y)
    save_model(m, "Heart", X_te, y_te, "heart_disease_model.sav")
except Exception as e:
    print(f"   ❌ Heart failed: {e}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 3. PARKINSON'S  — synthetic, UCI Oxford distribution
#    app.py input (plain list, 22 values, no column names):
#    [MDVP_Fo, MDVP_Fhi, MDVP_Flo, Jitter_pct, Jitter_Abs, RAP, PPQ,
#     Jitter_DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, Shimmer_DDA,
#     NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
# ══════════════════════════════════════════════════════════════════════════════
print("\n3️⃣  Training Parkinson's model …")
try:
    n_park = int(N * 0.75)
    n_heal = N - n_park
 
    def park_row(n_s, pd_positive):
        if pd_positive:
            return np.column_stack([
                np.random.normal(145, 30, n_s).clip(80, 260),  # Fo
                np.random.normal(188, 50, n_s).clip(100, 592), # Fhi
                np.random.normal(84, 25, n_s).clip(60, 240),   # Flo
                np.random.normal(0.006, 0.002, n_s).clip(0.001, 0.033),  # Jitter%
                np.random.normal(0.00005, 0.00003, n_s).clip(0, 0.00026),# Jitter Abs
                np.random.normal(0.0030, 0.001, n_s).clip(0, 0.021),     # RAP
                np.random.normal(0.0040, 0.001, n_s).clip(0, 0.019),     # PPQ
                np.random.normal(0.009, 0.003, n_s).clip(0, 0.063),      # DDP
                np.random.normal(0.040, 0.020, n_s).clip(0.009, 0.27),   # Shimmer
                np.random.normal(0.36, 0.15, n_s).clip(0.085, 2.28),     # Shimmer dB
                np.random.normal(0.020, 0.010, n_s).clip(0.004, 0.14),   # APQ3
                np.random.normal(0.030, 0.012, n_s).clip(0.006, 0.17),   # APQ5
                np.random.normal(0.030, 0.012, n_s).clip(0.007, 0.14),   # APQ
                np.random.normal(0.060, 0.030, n_s).clip(0.012, 0.43),   # DDA
                np.random.normal(0.025, 0.020, n_s).clip(0.0006, 0.31),  # NHR
                np.random.normal(20, 4, n_s).clip(8, 33),                 # HNR
                np.random.normal(0.50, 0.10, n_s).clip(0.26, 0.69),      # RPDE
                np.random.normal(0.82, 0.05, n_s).clip(0.57, 0.97),      # DFA
                np.random.normal(-5.5, 1.5, n_s).clip(-7.96, -2.43),     # spread1
                np.random.normal(0.22, 0.08, n_s).clip(0.006, 0.45),     # spread2
                np.random.normal(2.4, 0.5, n_s).clip(1.42, 3.67),        # D2
                np.random.normal(0.32, 0.08, n_s).clip(0.044, 0.53),     # PPE
            ])
        else:
            return np.column_stack([
                np.random.normal(197, 20, n_s).clip(150, 260),
                np.random.normal(223, 30, n_s).clip(160, 592),
                np.random.normal(170, 25, n_s).clip(100, 240),
                np.random.normal(0.0035, 0.001, n_s).clip(0.001, 0.009),
                np.random.normal(0.00002, 0.00001, n_s).clip(0, 0.00007),
                np.random.normal(0.0016, 0.0005, n_s).clip(0, 0.005),
                np.random.normal(0.0019, 0.0006, n_s).clip(0, 0.006),
                np.random.normal(0.0047, 0.001, n_s).clip(0, 0.015),
                np.random.normal(0.017, 0.007, n_s).clip(0.009, 0.045),
                np.random.normal(0.17, 0.06, n_s).clip(0.085, 0.42),
                np.random.normal(0.009, 0.003, n_s).clip(0.004, 0.021),
                np.random.normal(0.011, 0.004, n_s).clip(0.006, 0.027),
                np.random.normal(0.015, 0.005, n_s).clip(0.007, 0.030),
                np.random.normal(0.026, 0.010, n_s).clip(0.012, 0.063),
                np.random.normal(0.012, 0.006, n_s).clip(0.0006, 0.030),
                np.random.normal(24, 3, n_s).clip(15, 33),
                np.random.normal(0.42, 0.08, n_s).clip(0.26, 0.60),
                np.random.normal(0.74, 0.05, n_s).clip(0.57, 0.86),
                np.random.normal(-7.2, 1.2, n_s).clip(-7.96, -4.5),
                np.random.normal(0.16, 0.06, n_s).clip(0.006, 0.28),
                np.random.normal(2.0, 0.4, n_s).clip(1.42, 2.9),
                np.random.normal(0.19, 0.05, n_s).clip(0.044, 0.35),
            ])
 
    X = np.vstack([park_row(n_park, True), park_row(n_heal, False)])
    y = np.concatenate([np.ones(n_park, int), np.zeros(n_heal, int)])
    m, X_te, y_te = train_rf(X, y, stratify=True)
    save_model(m, "Parkinson's", X_te, y_te, "parkinsons_model.sav")
except Exception as e:
    print(f"   ❌ Parkinson's failed: {e}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 4. LIVER DISEASE  — synthetic, Indian Liver Patient Dataset distribution
#    app.py input (plain list, 10 values, no column names):
#    [Sex, age, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,
#     Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens,
#     Albumin, Albumin_and_Globulin_Ratio]
#    Sex: 0 = Male, 1 = Female
# ══════════════════════════════════════════════════════════════════════════════
print("\n4️⃣  Training Liver Disease model …")
try:
    sex      = np.random.choice([0,1], N, p=[0.75, 0.25])
    age      = np.random.normal(44, 16, N).clip(4, 90).astype(int)
    t_bil    = np.abs(np.random.exponential(1.5, N)).clip(0.4, 75.0)
    d_bil    = np.abs(np.random.exponential(0.6, N)).clip(0.1, 19.0)
    alk_phos = np.random.normal(290, 200, N).clip(63, 2110).astype(int)
    alt      = np.random.normal(80, 100, N).clip(10, 2000).astype(int)
    ast      = np.random.normal(80, 100, N).clip(10, 4929).astype(int)
    tot_prot = np.random.normal(6.5, 1.0, N).clip(2.7, 9.6)
    albumin  = np.random.normal(3.2, 0.8, N).clip(0.9, 5.5)
    ag_ratio = np.random.normal(0.95, 0.4, N).clip(0.3, 2.8)
 
    risk   = ((t_bil > 3)*0.40 + (albumin < 2.5)*0.30 + (alt > 200)*0.20
              + np.random.normal(0, 0.08, N))
    target = (risk > 0.35).astype(int)
 
    # Plain numpy array — matches app.py's plain list input
    X = np.column_stack([sex, age, t_bil, d_bil, alk_phos,
                         alt, ast, tot_prot, albumin, ag_ratio])
    y = target
    m, X_te, y_te = train_rf(X, y)
    save_model(m, "Liver", X_te, y_te, "liver_model.sav")
except Exception as e:
    print(f"   ❌ Liver failed: {e}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 5. HEPATITIS C  — synthetic, HCV dataset distribution
#    app.py sends a DataFrame with EXACT column names:
#    Age, Sex, ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT
#    Sex: 1 = Male, 2 = Female
# ══════════════════════════════════════════════════════════════════════════════
print("\n5️⃣  Training Hepatitis model …")
try:
    Age  = np.random.normal(47, 10, N).clip(19, 77).astype(int)
    Sex  = np.random.choice([1, 2], N, p=[0.67, 0.33])
    ALB  = np.random.normal(41, 7, N).clip(14, 82)
    ALP  = np.random.normal(68, 55, N).clip(11, 416)
    ALT  = np.random.normal(28, 55, N).clip(0.9, 325)
    AST  = np.random.normal(35, 45, N).clip(10, 324)
    BIL  = np.random.normal(11, 20, N).clip(0.8, 209)
    CHE  = np.random.normal(8.2, 2.8, N).clip(1.4, 16.4)
    CHOL = np.random.normal(5.4, 1.2, N).clip(1.4, 9.7)
    CREA = np.random.normal(81, 49, N).clip(8, 1079)
    GGT  = np.random.normal(39, 60, N).clip(4, 650)
    PROT = np.random.normal(72, 7, N).clip(44, 90)
 
    risk   = ((ALT > 80)*0.35 + (BIL > 25)*0.30 + (ALB < 35)*0.20
              + np.random.normal(0, 0.08, N))
    target = (risk > 0.35).astype(int)
 
    # Must use DataFrame with exact column names — matches app.py
    X = pd.DataFrame({
        'Age': Age, 'Sex': Sex, 'ALB': ALB, 'ALP': ALP,
        'ALT': ALT, 'AST': AST, 'BIL': BIL, 'CHE': CHE,
        'CHOL': CHOL, 'CREA': CREA, 'GGT': GGT, 'PROT': PROT
    })
    y = pd.Series(target)
    m, X_te, y_te = train_rf(X, y)
    save_model(m, "Hepatitis", X_te, y_te, "hepititisc_model.sav")
except Exception as e:
    print(f"   ❌ Hepatitis failed: {e}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 6. LUNG CANCER  — uses real lung_cancer.csv from project
#    app.py sends DataFrame with column names matching CSV headers
# ══════════════════════════════════════════════════════════════════════════════
print("\n6️⃣  Training Lung Cancer model …")
try:
    if not os.path.exists(LUNG_CSV):
        raise FileNotFoundError(
            f"lung_cancer.csv not found at: {LUNG_CSV}\n"
            "This file must already be in Frontend/data/")
 
    df = pd.read_csv(LUNG_CSV)
    df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]
 
    if df["GENDER"].dtype == object:
        df["GENDER"] = df["GENDER"].str.strip().map({"M": 1, "F": 2}).fillna(1)
 
    # Encode target separately: YES->1, NO->0  (clean binary classification)
    df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"YES": 1, "NO": 0})
 
    # Encode symptom columns: YES->2, NO->1  (matches app.py's yn() helper)
    symptom_cols = [c for c in df.columns if c not in ["GENDER", "AGE", "LUNG_CANCER"]]
    df[symptom_cols] = df[symptom_cols].replace({"YES": 2, "NO": 1})
 
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
 
    LUNG_COLS = ["GENDER","AGE","SMOKING","YELLOW_FINGERS","ANXIETY",
                 "PEER_PRESSURE","CHRONICDISEASE","FATIGUE","ALLERGY",
                 "WHEEZING","ALCOHOLCONSUMING","COUGHING",
                 "SHORTNESSOFBREATH","SWALLOWINGDIFFICULTY","CHESTPAIN"]
 
    X = df[LUNG_COLS]
    y = df["LUNG_CANCER"]    # now 0/1 integers
    m, X_te, y_te = train_rf(X, y)
    save_model(m, "Lung Cancer", X_te, y_te, "lung_cancer_model.sav")
except Exception as e:
    print(f"   ❌ Lung Cancer failed: {e}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 7. CHRONIC KIDNEY DISEASE  — synthetic, UCI CKD distribution
#    app.py sends DataFrame with EXACT column names:
#    age,bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,
#    pcv,wc,rc,htn,dm,cad,appet,pe,ane
# ══════════════════════════════════════════════════════════════════════════════
print("\n7️⃣  Training Chronic Kidney Disease model …")
try:
    age   = np.random.normal(51, 17, N).clip(2, 90).astype(int)
    bp    = np.random.normal(76, 13, N).clip(50, 180).astype(int)
    sg    = np.random.choice([1.005,1.010,1.015,1.020,1.025], N,
                              p=[0.10,0.15,0.20,0.35,0.20])
    al    = np.random.choice([0,1,2,3,4,5], N, p=[0.50,0.15,0.12,0.10,0.08,0.05])
    su    = np.random.choice([0,1,2,3,4,5], N, p=[0.60,0.15,0.10,0.07,0.05,0.03])
    rbc   = np.random.choice([1, 0], N, p=[0.65, 0.35])
    pc    = np.random.choice([1, 0], N, p=[0.55, 0.45])
    pcc   = np.random.choice([0, 1], N, p=[0.85, 0.15])
    ba    = np.random.choice([0, 1], N, p=[0.90, 0.10])
    bgr   = np.random.normal(148, 74, N).clip(22, 490).astype(int)
    bu    = np.random.normal(57, 50, N).clip(1.5, 391)
    sc    = np.random.normal(3.1, 3.8, N).clip(0.4, 76)
    sod   = np.random.normal(137, 10, N).clip(4.5, 163)
    pot   = np.random.normal(4.6, 1.0, N).clip(2.5, 47)
    hemo  = np.random.normal(12.5, 2.8, N).clip(3.1, 17.8)
    pcv   = np.random.normal(38, 9, N).clip(9, 54).astype(int)
    wc    = np.random.normal(8300, 3200, N).clip(2200, 26400).astype(int)
    rc    = np.random.normal(4.7, 1.0, N).clip(2.1, 8.0)
    htn   = np.random.binomial(1, 0.45, N)
    dm    = np.random.binomial(1, 0.35, N)
    cad   = np.random.binomial(1, 0.10, N)
    appet = np.random.choice([1, 0], N, p=[0.65, 0.35])
    pe    = np.random.binomial(1, 0.25, N)
    ane   = np.random.binomial(1, 0.30, N)
 
    risk   = ((sc > 5)*0.35 + (hemo < 10)*0.25 + (al > 2)*0.20
              + (rbc == 0)*0.10 + np.random.normal(0, 0.08, N))
    target = (risk > 0.35).astype(int)
 
    # Must use DataFrame with exact column names — matches app.py
    X = pd.DataFrame({
        'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
        'rbc': rbc, 'pc': pc, 'pcc': pcc, 'ba': ba, 'bgr': bgr,
        'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot, 'hemo': hemo,
        'pcv': pcv, 'wc': wc, 'rc': rc, 'htn': htn, 'dm': dm,
        'cad': cad, 'appet': appet, 'pe': pe, 'ane': ane
    })
    y = pd.Series(target)
    m, X_te, y_te = train_rf(X, y)
    save_model(m, "Kidney", X_te, y_te, "chronic_model.sav")
except Exception as e:
    print(f"   ❌ Kidney failed: {e}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 8. BREAST CANCER  — sklearn built-in, REAL data, app.py column names
#    FIX: sklearn uses "mean radius" but app.py uses "radius_mean"
#    Solution: load sklearn data, assign app.py column names when training
# ══════════════════════════════════════════════════════════════════════════════
print("\n8️⃣  Training Breast Cancer model …")
try:
    data = load_breast_cancer()
 
    # These EXACTLY match what app.py passes in its DataFrame
    APP_BREAST_COLS = [
        'radius_mean','texture_mean','perimeter_mean','area_mean',
        'smoothness_mean','compactness_mean','concavity_mean','concave points_mean',
        'symmetry_mean','fractal_dimension_mean',
        'radius_se','texture_se','perimeter_se','area_se',
        'smoothness_se','compactness_se','concavity_se','concave points_se',
        'symmetry_se','fractal_dimension_se',
        'radius_worst','texture_worst','perimeter_worst','area_worst',
        'smoothness_worst','compactness_worst','concavity_worst',
        'concave points_worst','symmetry_worst','fractal_dimension_worst'
    ]
 
    X = pd.DataFrame(data.data, columns=APP_BREAST_COLS)  # app.py names!
    y = pd.Series(data.target)
    m, X_te, y_te = train_rf(X, y, stratify=True)
    save_model(m, "Breast Cancer", X_te, y_te, "breast_cancer.sav")
except Exception as e:
    print(f"   ❌ Breast Cancer failed: {e}")
 
 
# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
saved = [f for f in os.listdir(MODELS_DIR) if f.endswith(".sav")]
print(f"✅ Done! {len(saved)}/8 models in {MODELS_DIR}:")
for f in sorted(saved):
    print(f"   {f}")
 
missing = {"diabetes_model.sav","heart_disease_model.sav","parkinsons_model.sav",
           "liver_model.sav","hepititisc_model.sav","lung_cancer_model.sav",
           "chronic_model.sav","breast_cancer.sav"} - set(saved)
if missing:
    print(f"\n⚠️  Still missing: {missing}")
else:
    print("\n🎉 All 8 models present! Run:  streamlit run app.py")
 
print()
print("⚠️  ONE manual fix still needed in app.py (~line 648):")
print("   The lung cancer model now returns 0/1 integers, not 'YES'/'NO'.")
print("   Find:    prediction = 1 if prediction_raw == \"YES\" else 0")
print("   Replace: prediction = int(prediction_raw)")
print("="*60)
