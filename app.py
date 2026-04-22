import streamlit as st
import base64
import pandas as pd
import pickle
import os
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
st.markdown("##### *Predictive Analysis based on Revised Atlanta Classification*")

# Load Model Assets
if not os.path.exists('model.pkl'):
    st.error("Model file not found.")
    st.stop()

with open('model.pkl', 'rb') as f:
    assets = pickle.load(f)

# --- 3. DYNAMIC INPUT GENERATION ---
user_data = {}
st.markdown("### 📋 Clinical Parameters")

# We separate 'Calculated' features to handle them specifically
calculated_features = ['SIRS', 'BISAP', 'BISAP Score'] 
base_features = [f for f in assets['features'] if not any(cf in f for cf in calculated_features)]

cols = st.columns(3)
for i, feature_name in enumerate(base_features):
    with cols[i % 3]:
        if feature_name in assets['le_dict']:
            user_data[feature_name] = st.selectbox(feature_name, assets['le_dict'][feature_name].classes_)
        elif "duration" in feature_name.lower():
            choice = st.selectbox(feature_name, ["Lesser than 3 days", "Greater than 3 days"])
            user_data[feature_name] = "1- 3 days" if "Lesser" in choice else "> 3 days"
        else:
            user_data[feature_name] = st.number_input(feature_name, min_value=0.0, value=0.0)

# --- 4. INTERACTIVE SCORING CALCULATOR ---
st.markdown("### 🧮 Clinical Scoring Calculator")
with st.expander("Calculate SIRS & BISAP Scores", expanded=True):
    sc1, sc2 = st.columns(2)
    
    with sc1:
        st.write("**SIRS Criteria**")
        # RR and PaCO2 are often not in the main feature set but needed for the score
        sirs_rr = st.checkbox("Respiratory Rate > 20/min OR PaCO2 < 32 mmHg")
        
        # Calculate SIRS automatically from existing inputs
        sirs_score = 0
        if user_data.get('Heart rate', 0) > 90: sirs_score += 1
        if user_data.get('Wbc count', 0) > 12000 or (user_data.get('Wbc count', 9999) < 4000 and user_data.get('Wbc count', 0) > 0): sirs_score += 1
        if sirs_rr: sirs_score += 1
        
        # Check Temp if it's a categorical selectbox
        temp_val = user_data.get('Temperature', '')
        if isinstance(temp_val, str) and ('> 38' in temp_val or '< 36' in temp_val): sirs_score += 1
        
        st.info(f"Calculated SIRS Score: **{sirs_score}** (SIRS Present: {'Yes' if sirs_score >= 2 else 'No'})")

    with sc2:
        st.write("**BISAP Components**")
        bisap_gcs = st.checkbox("Impaired Mental Status (GCS < 15)")
        
        # Calculate BISAP automatically
        bisap_val = 0
        if user_data.get('BUN', 0) > 25: bisap_val += 1
        if bisap_gcs: bisap_val += 1
        if sirs_score >= 2: bisap_val += 1
        if user_data.get('Age', 0) > 60: bisap_val += 1
        
        pe_val = user_data.get('Pleural effusion', '')
        if isinstance(pe_val, str) and 'yes' in pe_val.lower(): bisap_val += 1
        
        st.info(f"Calculated BISAP Score: **{bisap_val}**")

# Update user_data with the calculated scores for the model
for f in assets['features']:
    if 'SIRS' in f: user_data[f] = sirs_score
    if 'BISAP' in f: user_data[f] = bisap_val

# --- 5. PREDICTION ---
st.markdown("---")
if st.button("RUN ANALYSIS", use_container_width=True):
    # Map and Encode
    final_features = []
    for col in assets['features']:
        val = user_data.get(col, 0)
        if col in assets['le_dict']:
            try:
                encoded_val = assets['le_dict'][col].transform([str(val).strip()])[0]
                final_features.append(encoded_val)
            except:
                final_features.append(0)
        else:
            final_features.append(float(val))

    # Predict
    final_X = np.array([final_features])
    try:
        pred_idx = assets['model'].predict(final_X)[0]
        result = assets['le_target'].classes_[pred_idx]
        
        st.markdown(f'''
        <div class="glass-card" style="text-align: center; border-left: 10px solid #ff4b4b;">
            <h2 style="margin:0;">PREDICTED SEVERITY</h2>
            <h1 style="font-size: 3.5em; color: #ffeb3b !important;">{result.upper()}</h1>
            <p>Calculated using Random Forest Ensemble [cite: 63]</p>
        </div>
        ''', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {e}")
