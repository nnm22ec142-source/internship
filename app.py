import streamlit as st
import base64
import pandas as pd
import pickle
import os
import numpy as np

# --- 1. UI SETUP ---
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
            background: rgba(255, 255, 255, 0.1); border-radius: 20px;
            backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 30px; margin-top: 20px; text-align: center;
        }}
        h1, h2, h3, h5, p, label {{ color: white !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.7); }}
        </style>
    ''', unsafe_allow_html=True)

apply_custom_design('background.jpg')

# --- 2. ASSET LOADING ---
if not os.path.exists('model.pkl'):
    st.error("Model file not found.")
    st.stop()

with open('model.pkl', 'rb') as f:
    assets = pickle.load(f)

# --- 3. FRONTEND INPUTS ---
st.title("AP Severity Stratification Portal")
st.markdown("##### *Automated Decision Support System*")

# Filter out the calculated fields so they don't show up in the form
calc_fields = ['SIRS', 'BISAP', 'BISAP Score']
input_features = [f for f in assets['features'] if not any(cf in f for cf in calc_fields)]

user_data = {}
cols = st.columns(3)

# Display only the raw parameters
for i, feature_name in enumerate(input_features):
    with cols[i % 3]:
        if feature_name in assets['le_dict']:
            user_data[feature_name] = st.selectbox(feature_name, assets['le_dict'][feature_name].classes_)
        elif "duration" in feature_name.lower():
            choice = st.selectbox(feature_name, ["Lesser than 3 days", "Greater than 3 days"])
            user_data[feature_name] = "1- 3 days" if "Lesser" in choice else "> 3 days"
        else:
            user_data[feature_name] = st.number_input(feature_name, min_value=0.0, value=0.0)

# --- 4. BACKEND PROCESSING & PREDICTION ---
if st.button("RUN CLINICAL ANALYSIS", use_container_width=True):
    
    # --- A. BACKEND CALCULATION (SIRS) ---
    sirs_val = 0
    if user_data.get('Heart rate', 0) > 90: sirs_val += 1
    # Check WBC Count limits
    wbc = user_data.get('Wbc count', 5000)
    if wbc > 12000 or (0 < wbc < 4000): sirs_val += 1
    # Assuming Temperature is handled via selectbox or number
    temp = str(user_data.get('Temperature Status', ''))
    if "> 38" in temp or "< 36" in temp: sirs_val += 1
    # SIRS is positive if score >= 2
    sirs_present = 1 if sirs_val >= 2 else 0

    # --- B. BACKEND CALCULATION (BISAP) ---
    bisap_val = 0
    if user_data.get('BUN', 0) > 25: bisap_val += 1
    if sirs_present: bisap_val += 1
    if user_data.get('Age', 0) > 60: bisap_val += 1
    pe = str(user_data.get('Pleural effusion', '')).lower()
    if "yes" in pe or "present" in pe: bisap_val += 1
    # Mentation is usually a clinical observation input; if not present, defaults to 0
    
    # --- C. SYNCING FOR MODEL ---
    # Update the dictionary with the backend calculated values
    for f in assets['features']:
        if 'SIRS' in f: user_data[f] = sirs_val
        if 'BISAP' in f: user_data[f] = bisap_val

    # --- D. NUMERIC CONVERSION & PREDICTION ---
    final_features = []
    for col in assets['features']:
        val = user_data.get(col, 0)
        if col in assets['le_dict']:
            try:
                encoded = assets['le_dict'][col].transform([str(val).strip()])[0]
                final_features.append(encoded)
            except: final_features.append(0)
        else:
            # Numeric cleaner to prevent string-to-float error
            try:
                clean_val = "".join(c for c in str(val) if c.isdigit() or c == '.')
                final_features.append(float(clean_num) if clean_num else 0.0)
            except: final_features.append(float(val) if isinstance(val, (int, float)) else 0.0)

    try:
        final_X = np.array([final_features])
        pred_idx = assets['model'].predict(final_X)[0]
        result = assets['le_target'].classes_[pred_idx]
        
        st.markdown(f'''
        <div class="glass-card" style="border-left: 10px solid #ffeb3b;">
            <h2 style="margin:0;">FINAL SEVERITY RESULT</h2>
            <h1 style="color: #ffeb3b !important;">{result.upper()}</h1>
            <p style="font-size: 0.9em; opacity: 0.8;">Automated Backend Scoring: SIRS ({sirs_val}), BISAP ({bisap_val})</p>
        </div>
        ''', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
