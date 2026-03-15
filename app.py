import streamlit as st
import base64
import pandas as pd
import pickle
import os
import random

# --- 1. BACKGROUND & STYLE ---
def get_base64(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

def apply_custom_design(image_file):
    bin_str = get_base64(image_file)
    st.markdown(f'''
        <style>
        .stApp {{ background: none; }}
        .stApp::before {{
            content: "";
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover; background-attachment: fixed;
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            filter: blur(8px); z-index: -1; transform: scale(1.1);
        }}
        .glass-card {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px; backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 30px; margin-top: 20px;
        }}
        h1, h2, h3, h5, p, label {{ color: white !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.7); }}
        </style>
    ''', unsafe_allow_html=True)

apply_custom_design('background.jpg')

# --- 2. HEADER ---
col_t, col_l = st.columns([4, 1])
with col_t:
    st.title("Acute Pancreatitis Severity Stratification")
    st.markdown("##### *Predictive Analysis based on Revised Atlanta Classification*")
with col_l:
    if os.path.exists("logo.png"): st.image("logo.png", width=120)

# Load Model Assets
if not os.path.exists('model.pkl'):
    st.error("Model file not found. Please run train.py first.")
    st.stop()

with open('model.pkl', 'rb') as f:
    assets = pickle.load(f)

# --- 3. RANDOM VALUE INITIALIZER ---
def generate_random_defaults(assets):
    return {
        'Age': random.randint(18, 85),
        'Sex_idx': random.randint(0, len(assets['le_dict']['Sex'].classes_)-1),
        'Temperature_idx': random.randint(0, len(assets['le_dict']['Temperature'].classes_)-1),
        'BMI': round(random.uniform(18.5, 35.0), 1),
        'Etiology_idx': random.randint(0, len(assets['le_dict']['Etiology'].classes_)-1),
        'Heart rate': random.randint(60, 120),
        'Duration_idx': random.randint(0, 1),
        'SBP': random.randint(90, 160),
        'Wbc count': random.randint(4000, 20000),
        'Blood glucose': round(random.uniform(70.0, 250.0), 1),
        'Calcium': round(random.uniform(7.0, 10.5), 1),
        'BUN': round(random.uniform(10.0, 50.0), 1),
        'S. Creat': round(random.uniform(0.5, 3.5), 1),
        'S albumin': round(random.uniform(2.0, 5.0), 1),
        'S amylase': round(random.uniform(100.0, 3000.0), 1),
        'S. Lipase': round(random.uniform(100.0, 3000.0), 1),
        'Crp': round(random.uniform(5.0, 200.0), 1),
        'Pleural effusion_idx': random.randint(0, 1),
        'Platelet': random.randint(150000, 450000),
    }

# Only generate random defaults once; regenerate only when button is clicked
if 'rand' not in st.session_state:
    st.session_state.rand = generate_random_defaults(assets)

if st.button("🔀 GENERATE NEW RANDOM TEST DATA"):
    st.session_state.rand = generate_random_defaults(assets)
    st.rerun()

r = st.session_state.rand  # shorthand reference

st.markdown("### 📋 Clinical Parameters")
user_data = {}

# Section 1: Demographics & Vitals
with st.expander("🩺 Patient Vitals & Demographics", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        user_data['Age'] = st.number_input("Age", min_value=0, value=r['Age'])
        user_data['Sex'] = st.selectbox("Sex", assets['le_dict']['Sex'].classes_, index=r['Sex_idx'])
        user_data['Temperature'] = st.selectbox("Temperature Status", assets['le_dict']['Temperature'].classes_, index=r['Temperature_idx'])
    with c2:
        user_data['BMI'] = st.number_input("BMI", min_value=0.0, value=r['BMI'])
        user_data['Etiology'] = st.selectbox("Etiology", assets['le_dict']['Etiology'].classes_, index=r['Etiology_idx'])
        user_data['Heart rate'] = st.number_input("Heart Rate (bpm)", min_value=0, value=r['Heart rate'])
    with c3:
        durations = ["Lesser than 3 days", "Greater than 3 days"]
        duration_choice = st.selectbox("Duration of Symptoms", durations, index=r['Duration_idx'])
        user_data['Duration of symptoms'] = "1- 3 days" if "Lesser" in duration_choice else "> 3 days"
        user_data['SBP'] = st.number_input("Systolic BP (mmHg)", min_value=0, value=r['SBP'])

# Section 2: Laboratory Results
with st.expander("🧪 Laboratory Results", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        user_data['Wbc count'] = st.number_input("WBC Count", min_value=0, value=r['Wbc count'])
        user_data['Blood glucose'] = st.number_input("Blood Glucose", min_value=0.0, value=r['Blood glucose'])
        user_data['Calcium'] = st.number_input("Calcium (mg/dL)", min_value=0.0, value=r['Calcium'])
    with c2:
        user_data['BUN'] = st.number_input("BUN (mg/dL)", min_value=0.0, value=r['BUN'])
        user_data['S. Creat'] = st.number_input("S. Creatinine", min_value=0.0, value=r['S. Creat'])
        user_data['S albumin'] = st.number_input("S. Albumin (g/dL)", min_value=0.0, value=r['S albumin'])
    with c3:
        user_data['S amylase'] = st.number_input("S. Amylase (U/L)", min_value=0.0, value=r['S amylase'])
        user_data['S. Lipase'] = st.number_input("S. Lipase (U/L)", min_value=0.0, value=r['S. Lipase'])
        user_data['Crp'] = st.number_input("CRP (mg/L)", min_value=0.0, value=r['Crp'])

# Section 3: Imaging & Others
with st.expander("🛡️ Imaging & Others"):
    c1, c2 = st.columns(2)
    with c1:
        user_data['Pleural effusion'] = st.selectbox("Pleural Effusion", assets['le_dict']['Pleural effusion'].classes_, index=r['Pleural effusion_idx'])
    with c2:
        user_data['Platelet'] = st.number_input("Platelet Count", min_value=0, value=r['Platelet'])

# --- 4. PREDICTION LOGIC ---
if st.button("RUN ANALYSIS", use_container_width=True):
    # 1. Critical Field Check
    vitals_to_check = ['Age', 'BMI', 'Heart rate', 'Wbc count', 'Calcium', 'S amylase']
    zeros = [k for k in vitals_to_check if user_data.get(k, 0) <= 0]
    
    if zeros:
        st.error(f"⚠️ **Invalid Input:** The following cannot be zero: {', '.join(zeros)}")
    else:
        # 2. Create DataFrame
        input_df = pd.DataFrame([user_data])
        
        # 3. Ensure all columns the model expects are present
        for col in assets['features']:
            if col not in input_df.columns:
                input_df[col] = 0.0
        
        # 4. FIX: Force Encoding for Categorical Columns
        final_features = []
        for col in assets['features']:
            val = input_df[col].iloc[0]
            
            if col in assets['le_dict']:
                text_val = str(val).strip()
                try:
                    if text_val in assets['le_dict'][col].classes_:
                        encoded_val = assets['le_dict'][col].transform([text_val])[0]
                    else:
                        encoded_val = 0
                    final_features.append(encoded_val)
                except:
                    final_features.append(0)
            else:
                try:
                    final_features.append(float(val))
                except:
                    final_features.append(0.0)

        # 5. Create the final numeric array for the model
        import numpy as np
        final_X = np.array([final_features])
        
        # 6. Predict
        try:
            pred_idx = assets['model'].predict(final_X)[0]
            result = assets['le_target'].classes_[pred_idx]
            
            st.markdown(f'''
            <div class="glass-card" style="text-align: center; border-left: 10px solid #ff4b4b;">
                <h2 style="margin:0;">PREDICTED SEVERITY</h2>
                <h1 style="font-size: 3.5em; color: #ffeb3b !important;">{result.upper()}</h1>
                <p>Analysis based on Random Forest Clinical Modeling</p>
            </div>
            ''', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction Error: {e}")