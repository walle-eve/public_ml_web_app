"""
Kidney Disease Prediction System
"""

import pickle
from typing import Dict, Tuple, Any
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Constants
MODEL_PATH = 'kidney.pkl'
COLUMN_ORDER = [
    'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
    'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
    'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
    'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count',
    'red_blood_cell_count', 'hypertension', 'diabetes_mellitus',
    'coronary_artery_disease', 'appetite', 'peda_edema', 'aanemia'
]

class KidneyDiseasePredictor:
    def __init__(self):
        self.model = None
        self.le = LabelEncoder()
        self._load_model()
    
    def _load_model(self) -> None:
        try:
            with open(MODEL_PATH, 'rb') as file:
                self.model = pickle.load(file)
            if not isinstance(self.model, RandomForestClassifier):
                raise ValueError("Loaded model is not a RandomForestClassifier")
        except FileNotFoundError:
            st.error("Model file not found. Please ensure 'kidney.pkl' exists.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        try:
            categorical_features = {
                'red_blood_cells': input_data['red_blood_cells'],
                'pus_cell': input_data['pus_cell'],
                'pus_cell_clumps': input_data['pus_cell_clumps'],
                'bacteria': input_data['bacteria'],
                'hypertension': input_data['hypertension'],
                'diabetes_mellitus': input_data['diabetes_mellitus'],
                'coronary_artery_disease': input_data['coronary_artery_disease'],
                'appetite': input_data['appetite'],
                'peda_edema': input_data['peda_edema'],
                'aanemia': input_data['aanemia']
            }
            
            for feature, value in categorical_features.items():
                input_data[feature] = self.le.fit_transform([value])[0]
            
            return pd.DataFrame([input_data])[COLUMN_ORDER]
        except Exception as e:
            st.error(f"Error preprocessing input: {str(e)}")
            st.stop()
    
    def predict(self, input_data: Dict[str, Any]) -> Tuple[str, float, str]:
        try:
            processed_data = self.preprocess_input(input_data)
            prediction = self.model.predict(processed_data)[0]
            proba = self.model.predict_proba(processed_data)[0]
            
            if prediction == 0:
                return "CKD Positive 🚨", proba[0] * 100, "#E27D60"
            return "CKD Negative ✅", proba[1] * 100, "#50808E"
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.stop()

def setup_ui() -> None:
    st.set_page_config(
        layout="wide",
        page_title="KidneyDiseasePredictor",
        page_icon="https://i.imgur.com/nJZnxpE.png",
    )
    
    st.markdown("""
    <style>
    .stApp {
        background-color: #7A9D7E;
    }
    h1 {
        color: #E27D60;
        font-family: 'Arial', sans-serif;
        font-size: 36px;
        text-align: center;
        margin-bottom: 10px;
        margin-top: -20px;
    }
    .sidebar .sidebar-content {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
    }
    .stNumberInput, .stSelectbox {
        min-height: 82px;
    }
    .stButton button {
        background-color: #E27D60;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 20px;
        width: 100%;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid;
        text-align: center;
        margin-top: 20px;
    }
    div[data-testid="column"] {
        gap: 0.5rem;
    }
    .sidebar-text-box {
        background-color: #FAF3E0;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .sidebar-text-box a {
        color: #1E88E5;
        text-decoration: none;
    }
    .sidebar-text-box a:hover {
        text-decoration: underline;
    }
    </style>
    """, unsafe_allow_html=True)

def display_sidebar() -> None:
    st.sidebar.markdown("""
    <div style="text-align: center;">
        <img src="https://i.imgur.com/nJZnxpE.png" alt="Kidney Icon" width="50">
        <h1>KidneyDiseasePredictor</h1>
    </div>
    """, unsafe_allow_html=True)

    menu_option = st.sidebar.radio(
        "Menu",
        ["Instructions", "Advice", "Kidney Hospitals in Bangladesh"],
        index=0,
    )

    if menu_option == "Instructions":
        st.sidebar.markdown("""
        <div class="sidebar-text-box">
            <p><strong>Instructions:</strong></p>
            <p>- Fill in all the patient's details.</p>
            <p>- Click on the <strong>Kidney's Test Result</strong> button.</p>
            <p>- Review the prediction results.</p>
        </div>
        """, unsafe_allow_html=True)
    elif menu_option == "Advice":
        st.sidebar.markdown("""
        <div class="sidebar-text-box">
            <p><strong>Kidney Health Advice:</strong></p>
            <p>- Stay hydrated with plenty of water</p>
            <p>- Maintain a balanced, low-salt diet</p>
            <p>- Exercise regularly</p>
            <p>- Monitor blood pressure and sugar levels</p>
            <p>- Avoid smoking and limit alcohol</p>
        </div>
        """, unsafe_allow_html=True)
    elif menu_option == "Kidney Hospitals in Bangladesh":
        st.sidebar.markdown(
            """
            <div class="sidebar-text-box">
                <p><strong>Kidney Hospitals in Bangladesh:</strong></p>
                <p>1. <strong>Kidney Foundation Hospital And Research Institute</strong></p>
                <p>   - Located in Mirpur, Dhaka.</p>
                <p>   - Offers dialysis and transplantation services.</p>
                <p>   - <a href="https://kidneyfoundationbd.com" target="_blank">Website</a></p>
                <p>2. <strong>National Institute of Kidney Diseases & Urology</strong></p>
                <p>   - Government institute for kidney and urological diseases.</p>
                <p>   - <a href="https://nikdu.org.bd" target="_blank">Website</a></p>
                <p>3. <strong>Kidney Hospital</strong></p>
                <p>   - Multi-disciplinary hospital in Panthapath, Dhaka.</p>
                <p>   - <a href="https://dgkhl.com" target="_blank">Website</a></p>
                <p>4. <strong>Insaf Barakah Kidney & General Hospital</strong></p>
                <p>   - Specializes in urology and general healthcare.</p>
                <p>   - Located in Moghbazar, Dhaka.</p>
                <p>   - <a href="https://insafbarakahospital.com/Home/Contact" target="_blank">Website</a></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

def get_user_input() -> Dict[str, Any]:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=0)
        red_blood_cells = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
        blood_urea = st.number_input("Blood Urea (mg/dL)", min_value=0)
        packed_cell_volume = st.number_input("Packed Cell Volume (%)", min_value=0)
        coronary_artery_disease = st.selectbox("Coronary Artery Disease", ["No", "Yes"])
    
    with col2:
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0)
        pus_cell = st.selectbox("Pus Cells", ["Normal", "Abnormal"])
        serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, step=0.1, format="%.1f")
        white_blood_cell_count = st.number_input("White Blood Cells (cells/mm³)", min_value=0)
        appetite = st.selectbox("Appetite", ["Good", "Poor"])
    
    with col3:
        specific_gravity = st.number_input("Specific Gravity", min_value=0.0, max_value=2.0, value=1.000, step=0.001, format="%.3f")
        pus_cell_clumps = st.selectbox("Pus Cell Clumps", ["Not Present", "Present"])
        sodium = st.number_input("Sodium (mEq/L)", min_value=0)
        red_blood_cell_count = st.number_input("RBC Count (million cells/mm³)", min_value=0.0, value=0.0, step=0.1, format="%.1f")
        peda_edema = st.selectbox("Pedal Edema", ["No", "Yes"])
    
    with col4:
        albumin = st.number_input("Albumin (g/dL)", min_value=0)
        bacteria = st.selectbox("Bacteria", ["Not Present", "Present"])
        potassium = st.number_input("Potassium (mEq/L)", min_value=0.0, value=0.0, step=0.1, format="%.1f")
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        aanemia = st.selectbox("Anemia", ["No", "Yes"])
    
    with col5:
        sugar = st.number_input("Sugar (g/dL)", min_value=0)
        blood_glucose_random = st.number_input("Blood Glucose Random (mg/dL)", min_value=0)
        haemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, step=0.1, format="%.1f")
        diabetes_mellitus = st.selectbox("Diabetes Mellitus", ["No", "Yes"])

    return {
        "age": age,
        "blood_pressure": blood_pressure,
        "specific_gravity": specific_gravity,
        "albumin": albumin,
        "sugar": sugar,
        "red_blood_cells": red_blood_cells,
        "pus_cell": pus_cell,
        "pus_cell_clumps": pus_cell_clumps,
        "bacteria": bacteria,
        "blood_glucose_random": blood_glucose_random,
        "blood_urea": blood_urea,
        "serum_creatinine": serum_creatinine,
        "sodium": sodium,
        "potassium": potassium,
        "haemoglobin": haemoglobin,
        "packed_cell_volume": packed_cell_volume,
        "white_blood_cell_count": white_blood_cell_count,
        "red_blood_cell_count": red_blood_cell_count,
        "hypertension": hypertension,
        "diabetes_mellitus": diabetes_mellitus,
        "coronary_artery_disease": coronary_artery_disease,
        "appetite": appetite,
        "peda_edema": peda_edema,
        "aanemia": aanemia
    }

def main():
    setup_ui()
    display_sidebar()
    
    st.markdown('<h1 style="text-align: center;">Kidney Disease Prediction</h1>', unsafe_allow_html=True)
    
    predictor = KidneyDiseasePredictor()
    user_input = get_user_input()
    
    if st.button("Kidney's Test Result"):
        with st.spinner("Analyzing..."):
            label, confidence, color = predictor.predict(user_input)
            
            st.markdown(
                f"""
                <div class="result-box" style="border-color: {color};">
                    <h2 style="color: {color};">{label}</h2>
                    <p>Confidence: {confidence:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if label.startswith("CKD Positive"):
                st.warning("⚠️ Warning: Possible kidney disease detected. Please consult a doctor.")
    
    st.markdown("---")
    st.markdown('<p style="text-align: center;">Note: This application is for educational purposes only.</p>',
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()