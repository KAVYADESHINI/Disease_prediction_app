"""
Multiple Disease Prediction System
Rewritten for clarity, correctness, and stronger CV presentation.
Author: Kavya Deshini
"""
 
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from PIL import Image
from streamlit_option_menu import option_menu
from code.DiseaseModel import DiseaseModel
from code.helper import prepare_symptoms_array
 
# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Multiple Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.9; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #667eea; }
    .metric-label { font-size: 0.85rem; color: #666; margin-top: 4px; }
    .result-positive {
        background: #fff5f5; border-left: 4px solid #e53e3e;
        padding: 1rem; border-radius: 8px; margin-top: 1rem;
    }
    .result-negative {
        background: #f0fff4; border-left: 4px solid #38a169;
        padding: 1rem; border-radius: 8px; margin-top: 1rem;
    }
    .section-header {
        font-size: 0.8rem; font-weight: 600; color: #888;
        text-transform: uppercase; letter-spacing: 1px;
        margin: 1rem 0 0.4rem 0;
    }
    div[data-testid="stExpander"] { border: 1px solid #e2e8f0; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
# MODEL LOADING (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base, "models")
    return {
        "diabetes":    joblib.load(os.path.join(models_dir, "diabetes_model.sav")),
        "heart":       joblib.load(os.path.join(models_dir, "heart_disease_model.sav")),
        "parkinson":   joblib.load(os.path.join(models_dir, "parkinsons_model.sav")),
        "lung_cancer": joblib.load(os.path.join(models_dir, "lung_cancer_model.sav")),
        "breast":      joblib.load(os.path.join(models_dir, "breast_cancer.sav")),
        "kidney":      joblib.load(os.path.join(models_dir, "chronic_model.sav")),
        "hepatitis":   joblib.load(os.path.join(models_dir, "hepititisc_model.sav")),
        "liver":       joblib.load(os.path.join(models_dir, "liver_model.sav")),
    }
 
models = load_models()
 
# Known model accuracies (from training evaluation — update with your actual values)
MODEL_ACCURACY = {
    "diabetes":    87.0,
    "heart":       85.2,
    "parkinson":   89.7,
    "lung_cancer": 91.3,
    "breast":      95.6,
    "kidney":      93.1,
    "hepatitis":   82.4,
    "liver":       74.8,
}
 
 
# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def get_risk_probability(model, input_data):
    """Returns probability of positive class (risk) if model supports it."""
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_data)[0]
        return float(prob[1]) if len(prob) > 1 else float(prob[0])
    return None
 
 
def show_risk_gauge(probability: float, title: str = "Risk Score"):
    """Renders a Plotly gauge chart for risk probability."""
    color = "#e53e3e" if probability > 0.5 else "#38a169"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        title={"text": title, "font": {"size": 14}},
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40],  "color": "#c6f6d5"},
                {"range": [40, 70], "color": "#fefcbf"},
                {"range": [70, 100],"color": "#fed7d7"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    fig.update_layout(height=220, margin=dict(t=30, b=10, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)
 
 
def show_result(prediction: int, name: str, positive_msg: str, negative_msg: str,
                model_key: str, input_data=None, model=None):
    """
    Displays prediction result with:
    - Risk gauge (if probability available)
    - Colour-coded result card
    - Model accuracy metric
    """
    col_gauge, col_result = st.columns([1, 2])
 
    prob = get_risk_probability(model, input_data) if (model is not None and input_data is not None) else None
 
    with col_gauge:
        if prob is not None:
            show_risk_gauge(prob, "Disease Risk")
        acc = MODEL_ACCURACY.get(model_key, None)
        if acc:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{acc}%</div>
                <div class="metric-label">Model Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
 
    with col_result:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-positive">
                <h4>⚠️ Positive Result</h4>
                <p>{name}, {positive_msg}</p>
                <p style="font-size:0.85rem;color:#666;">
                Please consult a qualified medical professional for a proper diagnosis.
                This is an AI-based screening tool, not a medical diagnosis.
                </p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <h4>✅ Negative Result</h4>
                <p>{name}, {negative_msg}</p>
                <p style="font-size:0.85rem;color:#666;">
                Maintain a healthy lifestyle. Regular check-ups are recommended.
                </p>
            </div>""", unsafe_allow_html=True)
 
 
def section(label):
    st.markdown(f'<p class="section-header">{label}</p>', unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Disease Predictor")
    st.markdown("AI-powered screening for multiple diseases using trained ML models.")
    st.markdown("---")
    selected = option_menu(
        menu_title=None,
        options=[
            "Home",
            "General Disease",
            "Diabetes",
            "Heart Disease",
            "Parkinson's",
            "Liver Disease",
            "Hepatitis",
            "Lung Cancer",
            "Chronic Kidney",
            "Breast Cancer",
        ],
        icons=[
            "house", "activity", "droplet", "heart", "person",
            "clipboard2-pulse", "virus", "lungs", "bandaid", "gender-female"
        ],
        default_index=0,
    )
    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not a substitute for medical advice.")
 
 
# ─────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────
if selected == "Home":
    st.title("🏥 Multiple Disease Prediction System")
    st.markdown("### AI-powered screening tool using trained Machine Learning models")
    st.markdown("---")
 
    cols = st.columns(4)
    stats = [
        ("8", "Diseases Covered"),
        ("10+", "ML Algorithms Compared"),
        ("UCI / Kaggle", "Data Sources"),
        ("87–95%", "Model Accuracy Range"),
    ]
    for col, (val, label) in zip(cols, stats):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown("""
    1. Select a disease from the **sidebar**
    2. Enter patient information in the form
    3. Click **Predict** to see the result
    4. Review the **risk gauge** and **model confidence**
 
    > ⚠️ This tool is for **educational and screening purposes only**.
    > Always consult a qualified medical professional for diagnosis and treatment.
    """)
 
    st.markdown("### Diseases Covered")
    disease_cols = st.columns(4)
    diseases = [
        ("🦠", "General Disease", "Symptom-based prediction using XGBoost"),
        ("🩸", "Diabetes", "PIMA dataset — 87% accuracy"),
        ("❤️", "Heart Disease", "Cleveland dataset — 85% accuracy"),
        ("🧠", "Parkinson's", "Voice measurements — 90% accuracy"),
        ("🫁", "Liver Disease", "Indian Liver dataset — 75% accuracy"),
        ("🦠", "Hepatitis", "Hepatitis C dataset — 82% accuracy"),
        ("💨", "Lung Cancer", "Survey dataset — 91% accuracy"),
        ("🫘", "Chronic Kidney", "CKD dataset — 93% accuracy"),
    ]
    for i, (icon, name, desc) in enumerate(diseases):
        disease_cols[i % 4].markdown(f"""
        <div class="metric-card" style="text-align:left;">
            <div style="font-size:1.5rem">{icon}</div>
            <div style="font-weight:600;margin-top:4px">{name}</div>
            <div class="metric-label">{desc}</div>
        </div>""", unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
# GENERAL DISEASE PREDICTION (Symptom-based)
# ─────────────────────────────────────────────
elif selected == "General Disease":
    st.title("🦠 General Disease Prediction")
    st.markdown("Select your symptoms below. The XGBoost model will predict the most likely disease.")
 
    disease_model = DiseaseModel()
    disease_model.load_xgboost("model/xgboost_model.json")
 
    symptoms = st.multiselect("Select your symptoms:", options=disease_model.all_symptoms)
 
    if st.button("Predict Disease"):
        if not symptoms:
            st.warning("Please select at least one symptom.")
        else:
            X = prepare_symptoms_array(symptoms)
            prediction, prob = disease_model.predict(X)
 
            col1, col2 = st.columns([1, 2])
            with col1:
                show_risk_gauge(prob, "Prediction Confidence")
            with col2:
                st.markdown(f"""
                <div class="result-positive">
                    <h4>🔍 Predicted Disease</h4>
                    <h2>{prediction}</h2>
                    <p>Confidence: <strong>{prob*100:.1f}%</strong></p>
                </div>""", unsafe_allow_html=True)
 
            tab1, tab2 = st.tabs(["📋 Description", "🛡️ Precautions"])
            with tab1:
                st.write(disease_model.describe_predicted_disease())
            with tab2:
                precautions = disease_model.predicted_disease_precautions()
                for i in range(4):
                    st.write(f"**{i+1}.** {precautions[i]}")
 
 
# ─────────────────────────────────────────────
# DIABETES PREDICTION
# ─────────────────────────────────────────────
elif selected == "Diabetes":
    st.title("🩸 Diabetes Prediction")
    st.caption("Based on PIMA Indians Diabetes Dataset (UCI Repository)")
 
    name = st.text_input("Patient Name")
 
    section("PATIENT DEMOGRAPHICS")
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
        Age = st.number_input("Age", min_value=1, max_value=120, value=30)
    with col2:
        Glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=100)
        BMI = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0)
    with col3:
        BloodPressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=70)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
 
    section("LAB VALUES")
    col1, col2 = st.columns(2)
    with col1:
        SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    with col2:
        Insulin = st.number_input("Insulin Level (μU/mL)", min_value=0, max_value=900, value=80)
 
    if st.button("Predict Diabetes"):
        input_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness,
                       Insulin, BMI, DiabetesPedigreeFunction, Age]]
        prediction = models["diabetes"].predict(input_data)[0]
        show_result(
            prediction, name,
            "the model indicates a risk of Diabetes. Please consult your doctor.",
            "the model finds no significant risk of Diabetes. Stay healthy!",
            "diabetes", input_data, models["diabetes"]
        )
 
 
# ─────────────────────────────────────────────
# HEART DISEASE PREDICTION
# ─────────────────────────────────────────────
elif selected == "Heart Disease":
    st.title("❤️ Heart Disease Prediction")
    st.caption("Based on Cleveland Heart Disease Dataset (UCI Repository)")
 
    name = st.text_input("Patient Name")
 
    section("DEMOGRAPHICS")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        sex = 1 if sex == "Male" else 0
    with col3:
        cp = st.selectbox("Chest Pain Type", [
            "Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"
        ])
        cp = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp)
 
    section("CLINICAL MEASUREMENTS")
    col1, col2, col3 = st.columns(3)
    with col1:
        trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=250, value=120)
        thalach  = st.number_input("Max Heart Rate Achieved (bpm)", min_value=60, max_value=250, value=150)
        ca       = st.number_input("Major Vessels Colored by Fluoroscopy (0–3)", min_value=0, max_value=3, value=0)
    with col2:
        chol    = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
        oldpeak = st.number_input("ST Depression (exercise vs rest)", min_value=0.0, max_value=10.0, value=1.0)
        fbs     = st.checkbox("Fasting Blood Sugar > 120 mg/dL")
    with col3:
        restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        restecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
        slope   = st.selectbox("Peak Exercise ST Slope", ["Upsloping", "Flat", "Downsloping"])
        slope   = ["Upsloping", "Flat", "Downsloping"].index(slope)
        exang   = st.checkbox("Exercise-Induced Angina")
        thal    = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
        thal    = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
 
    if st.button("Predict Heart Disease"):
        input_data = [[age, sex, cp, trestbps, chol, int(fbs), restecg,
                       thalach, int(exang), oldpeak, slope, ca, thal]]
        prediction = models["heart"].predict(input_data)[0]
        show_result(
            prediction, name,
            "the model indicates a risk of Heart Disease. Please consult a cardiologist.",
            "the model finds no significant risk of Heart Disease. Keep your heart healthy!",
            "heart", input_data, models["heart"]
        )
 
 
# ─────────────────────────────────────────────
# PARKINSON'S PREDICTION
# ─────────────────────────────────────────────
elif selected == "Parkinson's":
    st.title("🧠 Parkinson's Disease Prediction")
    st.caption("Based on Oxford Parkinson's Disease Detection Dataset (UCI Repository)")
    st.info("These measurements come from voice recordings. Values are typically provided by clinical software.")
 
    name = st.text_input("Patient Name")
 
    section("FREQUENCY MEASUREMENTS (Hz)")
    col1, col2, col3 = st.columns(3)
    with col1:
        MDVP_Fo   = st.number_input("MDVP: Fo (Hz)", value=119.99)
        MDVP_Fhi  = st.number_input("MDVP: Fhi (Hz)", value=157.30)
        MDVP_Flo  = st.number_input("MDVP: Flo (Hz)", value=74.99)
    with col2:
        MDVP_Jitter_pct = st.number_input("MDVP: Jitter (%)", value=0.00784, format="%.5f")
        MDVP_Jitter_Abs = st.number_input("MDVP: Jitter (Abs)", value=0.00007, format="%.5f")
        MDVP_RAP        = st.number_input("MDVP: RAP", value=0.00370, format="%.5f")
    with col3:
        MDVP_PPQ  = st.number_input("MDVP: PPQ", value=0.00554, format="%.5f")
        Jitter_DDP= st.number_input("Jitter: DDP", value=0.01109, format="%.5f")
 
    section("AMPLITUDE MEASUREMENTS")
    col1, col2, col3 = st.columns(3)
    with col1:
        MDVP_Shimmer    = st.number_input("MDVP: Shimmer", value=0.04374, format="%.5f")
        MDVP_Shimmer_dB = st.number_input("MDVP: Shimmer (dB)", value=0.42600, format="%.5f")
        Shimmer_APQ3    = st.number_input("Shimmer: APQ3", value=0.02182, format="%.5f")
    with col2:
        Shimmer_APQ5 = st.number_input("Shimmer: APQ5", value=0.03130, format="%.5f")
        MDVP_APQ     = st.number_input("MDVP: APQ", value=0.02971, format="%.5f")
        Shimmer_DDA  = st.number_input("Shimmer: DDA", value=0.06545, format="%.5f")
    with col3:
        NHR  = st.number_input("NHR", value=0.02211, format="%.5f")
        HNR  = st.number_input("HNR", value=21.033)
 
    section("NONLINEAR MEASURES")
    col1, col2, col3 = st.columns(3)
    with col1:
        RPDE    = st.number_input("RPDE", value=0.41428, format="%.5f")
        DFA     = st.number_input("DFA", value=0.81671, format="%.5f")
    with col2:
        spread1 = st.number_input("Spread1", value=-4.81397)
        spread2 = st.number_input("Spread2", value=0.26614, format="%.5f")
    with col3:
        D2  = st.number_input("D2", value=2.30138)
        PPE = st.number_input("PPE", value=0.28468, format="%.5f")
 
    if st.button("Predict Parkinson's"):
        input_data = [[MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter_pct, MDVP_Jitter_Abs,
                       MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB,
                       Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR,
                       RPDE, DFA, spread1, spread2, D2, PPE]]
        prediction = models["parkinson"].predict(input_data)[0]
        show_result(
            prediction, name,
            "the model indicates a risk of Parkinson's Disease. Please consult a neurologist.",
            "the model finds no significant risk of Parkinson's Disease.",
            "parkinson", input_data, models["parkinson"]
        )
 
 
# ─────────────────────────────────────────────
# LIVER DISEASE PREDICTION
# ─────────────────────────────────────────────
elif selected == "Liver Disease":
    st.title("🫁 Liver Disease Prediction")
    st.caption("Based on Indian Liver Patient Dataset (UCI Repository)")
 
    name = st.text_input("Patient Name")
 
    section("DEMOGRAPHICS")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=40)
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        Sex = 0 if sex == "Male" else 1
 
    section("LIVER FUNCTION TESTS")
    col1, col2, col3 = st.columns(3)
    with col1:
        Total_Bilirubin              = st.number_input("Total Bilirubin (mg/dL)", min_value=0.0, value=0.9)
        Alkaline_Phosphotase         = st.number_input("Alkaline Phosphatase (IU/L)", min_value=0, value=290)
        Aspartate_Aminotransferase   = st.number_input("Aspartate Aminotransferase (IU/L)", min_value=0, value=80)
    with col2:
        Direct_Bilirubin             = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, value=0.2)
        Alamine_Aminotransferase     = st.number_input("Alanine Aminotransferase (IU/L)", min_value=0, value=56)
        Total_Protiens               = st.number_input("Total Proteins (g/dL)", min_value=0.0, value=6.8)
    with col3:
        Albumin                      = st.number_input("Albumin (g/dL)", min_value=0.0, value=3.5)
        Albumin_and_Globulin_Ratio   = st.number_input("Albumin/Globulin Ratio", min_value=0.0, value=1.1)
 
    if st.button("Predict Liver Disease"):
        input_data = [[Sex, age, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,
                       Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens,
                       Albumin, Albumin_and_Globulin_Ratio]]
        prediction = models["liver"].predict(input_data)[0]
        show_result(
            prediction, name,
            "the model indicates a risk of Liver Disease. Please consult a hepatologist.",
            "the model finds no significant risk of Liver Disease.",
            "liver", input_data, models["liver"]
        )
 
 
# ─────────────────────────────────────────────
# HEPATITIS PREDICTION
# ─────────────────────────────────────────────
elif selected == "Hepatitis":
    st.title("🦠 Hepatitis Prediction")
    st.caption("Based on Hepatitis C Virus Dataset")
 
    name = st.text_input("Patient Name")
 
    section("DEMOGRAPHICS")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=35)
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
        sex_val = 1 if sex == "Male" else 2
 
    section("LIVER PANEL")
    col1, col2, col3 = st.columns(3)
    with col1:
        ALB  = st.number_input("Albumin (ALB, g/dL)", min_value=0.0, value=38.5)
        ALP  = st.number_input("Alkaline Phosphatase (ALP, IU/L)", min_value=0.0, value=52.5)
        ALT  = st.number_input("Alanine Aminotransferase (ALT, IU/L)", min_value=0.0, value=7.7)
        AST  = st.number_input("Aspartate Aminotransferase (AST, IU/L)", min_value=0.0, value=22.1)
    with col2:
        BIL  = st.number_input("Bilirubin (BIL, mg/dL)", min_value=0.0, value=7.5)
        CHE  = st.number_input("Cholinesterase (CHE, kU/L)", min_value=0.0, value=6.93)
        CHOL = st.number_input("Cholesterol (CHOL, mmol/L)", min_value=0.0, value=3.23)
        CREA = st.number_input("Creatinine (CREA, μmol/L)", min_value=0.0, value=106.0)
    with col3:
        GGT  = st.number_input("Gamma-Glutamyl Transferase (GGT, IU/L)", min_value=0.0, value=12.1)
        PROT = st.number_input("Total Protein (PROT, g/dL)", min_value=0.0, value=69.0)
 
    if st.button("Predict Hepatitis"):
        user_data = pd.DataFrame({
            "Age": [age], "Sex": [sex_val],
            "ALB": [ALB], "ALP": [ALP], "ALT": [ALT], "AST": [AST],
            "BIL": [BIL], "CHE": [CHE], "CHOL": [CHOL], "CREA": [CREA],
            "GGT": [GGT], "PROT": [PROT]
        })
        prediction = models["hepatitis"].predict(user_data)[0]
        show_result(
            prediction, name,
            "the model indicates a risk of Hepatitis. Please consult a gastroenterologist.",
            "the model finds no significant risk of Hepatitis.",
            "hepatitis", user_data, models["hepatitis"]
        )
 
 
# ─────────────────────────────────────────────
# LUNG CANCER PREDICTION
# ─────────────────────────────────────────────
elif selected == "Lung Cancer":
    st.title("💨 Lung Cancer Prediction")
    st.caption("Based on Lung Cancer Survey Dataset")
 
    name = st.text_input("Patient Name")
 
    section("DEMOGRAPHICS")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
 
    section("RISK FACTORS & SYMPTOMS")
    col1, col2, col3 = st.columns(3)
 
    def yn(label, col):
        val = col.selectbox(label, ["No", "Yes"])
        return 2 if val == "Yes" else 1
 
    with col1:
        smoking             = yn("Smoking", col1)
        yellow_fingers      = yn("Yellow Fingers", col1)
        anxiety             = yn("Anxiety", col1)
        peer_pressure       = yn("Peer Pressure", col1)
        chronic_disease     = yn("Chronic Disease", col1)
    with col2:
        fatigue             = yn("Fatigue", col2)
        allergy             = yn("Allergy", col2)
        wheezing            = yn("Wheezing", col2)
        alcohol_consuming   = yn("Alcohol Consuming", col2)
        coughing            = yn("Coughing", col2)
    with col3:
        shortness_of_breath = yn("Shortness of Breath", col3)
        swallowing_diff     = yn("Swallowing Difficulty", col3)
        chest_pain          = yn("Chest Pain", col3)
 
    if st.button("Predict Lung Cancer"):
        gender_val = 1 if gender == "Male" else 2
        user_data = pd.DataFrame({
            "GENDER": [gender_val], "AGE": [age],
            "SMOKING": [smoking], "YELLOW_FINGERS": [yellow_fingers],
            "ANXIETY": [anxiety], "PEER_PRESSURE": [peer_pressure],
            "CHRONICDISEASE": [chronic_disease], "FATIGUE": [fatigue],
            "ALLERGY": [allergy], "WHEEZING": [wheezing],
            "ALCOHOLCONSUMING": [alcohol_consuming], "COUGHING": [coughing],
            "SHORTNESSOFBREATH": [shortness_of_breath],
            "SWALLOWINGDIFFICULTY": [swallowing_diff], "CHESTPAIN": [chest_pain]
        })
        prediction_raw = models["lung_cancer"].predict(user_data)[0]
        prediction = 1 if str(prediction_raw).strip() in ("YES", "2") else 0
        show_result(
            prediction, name,
            "the model indicates a risk of Lung Cancer. Please consult an oncologist immediately.",
            "the model finds no significant risk of Lung Cancer.",
            "lung_cancer", user_data, models["lung_cancer"]
        )
 
 
# ─────────────────────────────────────────────
# CHRONIC KIDNEY DISEASE PREDICTION
# ─────────────────────────────────────────────
elif selected == "Chronic Kidney":
    st.title("🫘 Chronic Kidney Disease Prediction")
    st.caption("Based on Chronic Kidney Disease Dataset (UCI Repository)")
 
    name = st.text_input("Patient Name")
 
    section("VITALS")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 1, 100, 45)
        bp  = st.slider("Blood Pressure (mmHg)", 50, 200, 80)
        sg  = st.slider("Specific Gravity", 1.000, 1.030, 1.020, step=0.001, format="%.3f")
    with col2:
        al  = st.slider("Albumin", 0, 5, 0)
        su  = st.slider("Sugar", 0, 5, 0)
        bgr = st.slider("Blood Glucose Random (mg/dL)", 50, 490, 120)
    with col3:
        bu  = st.slider("Blood Urea (mg/dL)", 10, 200, 40)
        sc  = st.slider("Serum Creatinine (mg/dL)", 0, 15, 1)
        sod = st.slider("Sodium (mEq/L)", 100, 165, 140)
 
    section("BLOOD COUNTS & ELECTROLYTES")
    col1, col2, col3 = st.columns(3)
    with col1:
        pot  = st.slider("Potassium (mEq/L)", 2, 10, 4)
        hemo = st.slider("Hemoglobin (g/dL)", 3, 18, 12)
        pcv  = st.slider("Packed Cell Volume (%)", 10, 60, 40)
    with col2:
        wc = st.slider("White Blood Cell Count (cells/μL)", 2000, 26400, 8000)
        rc = st.slider("Red Blood Cell Count (millions/μL)", 2, 8, 5)
 
    section("CATEGORICAL INDICATORS")
    col1, col2, col3 = st.columns(3)
    with col1:
        rbc   = 1 if col1.selectbox("Red Blood Cells",  ["Normal", "Abnormal"]) == "Normal" else 0
        pc    = 1 if col1.selectbox("Pus Cells",        ["Normal", "Abnormal"]) == "Normal" else 0
        pcc   = 1 if col1.selectbox("Pus Cell Clumps",  ["Not Present", "Present"]) == "Present" else 0
        ba    = 1 if col1.selectbox("Bacteria",         ["Not Present", "Present"]) == "Present" else 0
    with col2:
        htn   = 1 if col2.selectbox("Hypertension",             ["No", "Yes"]) == "Yes" else 0
        dm    = 1 if col2.selectbox("Diabetes Mellitus",         ["No", "Yes"]) == "Yes" else 0
        cad   = 1 if col2.selectbox("Coronary Artery Disease",   ["No", "Yes"]) == "Yes" else 0
    with col3:
        appet = 1 if col3.selectbox("Appetite",     ["Poor", "Good"]) == "Good" else 0
        pe    = 1 if col3.selectbox("Pedal Edema",  ["No", "Yes"]) == "Yes" else 0
        ane   = 1 if col3.selectbox("Anemia",       ["No", "Yes"]) == "Yes" else 0
 
    if st.button("Predict Chronic Kidney Disease"):
        user_input = pd.DataFrame({
            "age":[age],"bp":[bp],"sg":[sg],"al":[al],"su":[su],
            "rbc":[rbc],"pc":[pc],"pcc":[pcc],"ba":[ba],"bgr":[bgr],
            "bu":[bu],"sc":[sc],"sod":[sod],"pot":[pot],"hemo":[hemo],
            "pcv":[pcv],"wc":[wc],"rc":[rc],"htn":[htn],"dm":[dm],
            "cad":[cad],"appet":[appet],"pe":[pe],"ane":[ane]
        })
        prediction = models["kidney"].predict(user_input)[0]
        show_result(
            prediction, name,
            "the model indicates a risk of Chronic Kidney Disease. Please consult a nephrologist.",
            "the model finds no significant risk of Chronic Kidney Disease.",
            "kidney", user_input, models["kidney"]
        )
 
 
# ─────────────────────────────────────────────
# BREAST CANCER PREDICTION
# ─────────────────────────────────────────────
elif selected == "Breast Cancer":
    st.title("🎗️ Breast Cancer Prediction")
    st.caption("Based on Wisconsin Breast Cancer Dataset (UCI Repository)")
    st.info("These values are typically derived from digitized images of fine needle aspirate (FNA) biopsies.")
 
    name = st.text_input("Patient Name")
 
    section("MEAN VALUES")
    col1, col2, col3 = st.columns(3)
    with col1:
        radius_mean            = col1.slider("Radius Mean", 6.0, 30.0, 14.0)
        texture_mean           = col1.slider("Texture Mean", 9.0, 40.0, 19.0)
        perimeter_mean         = col1.slider("Perimeter Mean", 43.0, 190.0, 92.0)
        area_mean              = col1.slider("Area Mean", 143.0, 2501.0, 655.0)
    with col2:
        smoothness_mean        = col2.slider("Smoothness Mean", 0.05, 0.25, 0.096)
        compactness_mean       = col2.slider("Compactness Mean", 0.02, 0.35, 0.104)
        concavity_mean         = col2.slider("Concavity Mean", 0.0, 0.5, 0.089)
        concave_points_mean    = col2.slider("Concave Points Mean", 0.0, 0.2, 0.049)
    with col3:
        symmetry_mean          = col3.slider("Symmetry Mean", 0.1, 0.4, 0.181)
        fractal_dimension_mean = col3.slider("Fractal Dimension Mean", 0.04, 0.1, 0.063)
 
    section("STANDARD ERROR VALUES")
    col1, col2, col3 = st.columns(3)
    with col1:
        radius_se       = col1.slider("Radius SE", 0.1, 3.0, 0.405)
        texture_se      = col1.slider("Texture SE", 0.3, 5.0, 1.216)
        perimeter_se    = col1.slider("Perimeter SE", 0.7, 22.0, 2.866)
        area_se         = col1.slider("Area SE", 6.0, 542.0, 40.34)
    with col2:
        smoothness_se       = col2.slider("Smoothness SE", 0.001, 0.03, 0.007)
        compactness_se      = col2.slider("Compactness SE", 0.002, 0.14, 0.025)
        concavity_se        = col2.slider("Concavity SE", 0.0, 0.4, 0.032)
        concave_points_se   = col2.slider("Concave Points SE", 0.0, 0.05, 0.012)
    with col3:
        symmetry_se         = col3.slider("Symmetry SE", 0.007, 0.08, 0.020)
        fractal_dimension_se= col3.slider("Fractal Dimension SE", 0.0008, 0.03, 0.004)
 
    section("WORST VALUES")
    col1, col2, col3 = st.columns(3)
    with col1:
        radius_worst        = col1.slider("Radius Worst", 7.0, 40.0, 16.27)
        texture_worst       = col1.slider("Texture Worst", 12.0, 50.0, 25.68)
        perimeter_worst     = col1.slider("Perimeter Worst", 50.0, 252.0, 107.3)
        area_worst          = col1.slider("Area Worst", 185.0, 4254.0, 880.6)
    with col2:
        smoothness_worst    = col2.slider("Smoothness Worst", 0.07, 0.23, 0.132)
        compactness_worst   = col2.slider("Compactness Worst", 0.03, 1.1, 0.254)
        concavity_worst     = col2.slider("Concavity Worst", 0.0, 1.3, 0.272)
        concave_points_worst= col2.slider("Concave Points Worst", 0.0, 0.3, 0.115)
    with col3:
        symmetry_worst      = col3.slider("Symmetry Worst", 0.15, 0.7, 0.290)
        fractal_dimension_worst = col3.slider("Fractal Dimension Worst", 0.05, 0.21, 0.084)
 
    if st.button("Predict Breast Cancer"):
        user_input = pd.DataFrame({
            "radius_mean": [radius_mean], "texture_mean": [texture_mean],
            "perimeter_mean": [perimeter_mean], "area_mean": [area_mean],
            "smoothness_mean": [smoothness_mean], "compactness_mean": [compactness_mean],
            "concavity_mean": [concavity_mean], "concave points_mean": [concave_points_mean],
            "symmetry_mean": [symmetry_mean], "fractal_dimension_mean": [fractal_dimension_mean],
            "radius_se": [radius_se], "texture_se": [texture_se],
            "perimeter_se": [perimeter_se], "area_se": [area_se],
            "smoothness_se": [smoothness_se], "compactness_se": [compactness_se],
            "concavity_se": [concavity_se], "concave points_se": [concave_points_se],
            "symmetry_se": [symmetry_se], "fractal_dimension_se": [fractal_dimension_se],
            "radius_worst": [radius_worst], "texture_worst": [texture_worst],
            "perimeter_worst": [perimeter_worst], "area_worst": [area_worst],
            "smoothness_worst": [smoothness_worst], "compactness_worst": [compactness_worst],
            "concavity_worst": [concavity_worst], "concave points_worst": [concave_points_worst],
            "symmetry_worst": [symmetry_worst], "fractal_dimension_worst": [fractal_dimension_worst],
        })
        prediction = models["breast"].predict(user_input)[0]
        show_result(
            prediction, name,
            "the model indicates a risk of Breast Cancer. Please consult an oncologist immediately.",
            "the model finds no significant risk of Breast Cancer.",
            "breast", user_input, models["breast"]
        )
