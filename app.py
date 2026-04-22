import streamlit as st
import base64
import pandas as pd
import pickle
import os
import random
import numpy as np

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
st.title("Acute Pancreatitis Severity Stratification")
st.markdown("##### *Comprehensive Predictive Analysis (All Dataset Parameters)*")

# Load Model Assets
if not os.path.exists('model.pkl'):
    st.error("Model file not found.")
    st.stop()

with open('model.pkl', 'rb') as f:
    assets = pickle.load(f)

# --- 3. DYNAMIC INPUT GENERATION ---
# This dictionary will store all user inputs based on the Excel headers
user_data = {}

st.markdown("### 📋 Clinical Parameters (Full Dataset)")

# Create a 3-column layout to handle many inputs efficiently
cols = st.columns(3)

# Loop through every feature present in the trained model/Excel sheet
for i, feature_name in enumerate(assets['features']):
    with cols[i % 3]:
        # A. Handle Categorical Columns (if present in LabelEncoder dictionary)
        if feature_name in assets['le_dict']:
            options = assets['le_dict'][feature_name].classes_
            user_data[feature_name] = st.selectbox(
                f"{feature_name}", 
                options, 
                key=f"select_{feature_name}"
            )
        
        # B. Handle Duration (Special logic from your original code if needed)
        elif "duration" in feature_name.lower():
            durations = ["Lesser than 3 days", "Greater than 3 days"]
            choice = st.selectbox(feature_name, durations, key="dur_choice")
            user_data[feature_name] = "1- 3 days" if "Lesser" in choice else "> 3 days"
            
        # C. Handle Numeric Columns
        else:
            # Check if feature name suggests a whole number or float
            if any(x in feature_name.lower() for x in ['age', 'count', 'platelet', 'rate', 'sbp']):
                user_data[feature_name] = st.number_input(
                    feature_name, 
                    min_value=0, 
                    value=0, 
                    step=1,
                    key=f"num_{feature_name}"
                )
            else:
                user_data[feature_name] = st.number_input(
                    feature_name, 
                    min_value=0.0, 
                    value=0.0, 
                    format="%.2f",
                    key=f"num_{feature_name}"
                )

# --- 4. PREDICTION LOGIC ---
st.markdown("---")
if st.button("RUN ANALYSIS", use_container_width=True):
    # 1. Check for zeros in key vital fields (Adjust list as per your Excel headers)
    vitals_to_validate = [f for f in assets['features'] if any(v in f.lower() for v in ['age', 'bmi', 'calcium', 'amylase'])]
    zeros = [k for k in vitals_to_validate if user_data.get(k, 0) <= 0]
    
    if zeros:
        st.warning(f"Note: Some parameters are zero ({', '.join(zeros[:3])}...). Proceeding with analysis.")

    # 2. Preprocessing and Encoding
    final_features = []
    for col in assets['features']:
        val = user_data[col]
        
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

    # 3. Predict
    final_X = np.array([final_features])
    try:
        pred_idx = assets['model'].predict(final_X)[0]
        result = assets['le_target'].classes_[pred_idx]
        
        st.markdown(f'''
        <div class="glass-card" style="text-align: center; border-left: 10px solid #ff4b4b;">
            <h2 style="margin:0;">PREDICTED SEVERITY</h2>
            <h1 style="font-size: 3.5em; color: #ffeb3b !important;">{result.upper()}</h1>
            <p>Analysis based on All Dataset Parameters</p>
        </div>
        ''', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction Error: {e}")
