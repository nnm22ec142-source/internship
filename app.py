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
        .stExpander {{ background: rgba(0, 0, 0, 0.2); border-radius: 10px; }}
        </style>
    ''', unsafe_allow_html=True)

# Ensure you have a background.jpg in your directory
apply_custom_design('background.jpg')

# --- 2. HEADER & ASSET LOADING ---
col_t, col_l = st.columns([4, 1])
with col_t:
    st.title("Acute Pancreatitis Severity Stratification")
    st.markdown("##### *Predictive Analysis based on Revised Atlanta Classification*")
with col_l:
    if os.path.exists("logo.png"): st.image("logo.png", width=120)

if not os.path.exists('model.pkl'):
    st.error("Model file (model.pkl) not found. Please ensure it is in the root directory.")
    st.stop()

with open('model.pkl', 'rb') as f:
    assets = pickle.load(f)

# --- 3. DYNAMIC INPUT GENERATION ---
# Features to handle via calculator rather than manual input
calc_fields = ['SIRS', 'BISAP', 'BISAP Score']
# Filter out the calculated fields from the main loop
input_features = [f for f in assets['features'] if not any(cf in f for cf in calc_fields)]

user_data = {}

st.markdown("### 📋 Clinical Parameters")
cols = st.columns(3)

for i, feature_name in enumerate(input_features):
    with cols[i % 3]:
        # Handle Categorical Data
        if feature_name in assets['le_dict']:
            user_data[feature_name] = st.selectbox(
                feature_name, 
                assets['le_dict'][feature_name].classes_,
                key=f"select_{feature_name}"
            )
        # Handle Duration logic
        elif "duration" in feature_name.lower():
            choice = st.selectbox(feature_name, ["Lesser than 3 days", "Greater than 3 days"], key="dur")
            user_data[feature_name] = "1- 3 days" if "Lesser" in choice else "> 3 days"
        # Handle Numeric Data
        else:
            # Detect if it should be an integer or float
            if any(x in feature_name.lower() for x in ['age', 'count', 'platelet', 'rate', 'sbp']):
                user_data[feature_name] = st.number_input(feature_name, min_value=0, value=0, step=1, key=f"num_{feature_name}")
            else:
                user_data[feature_name] = st.number_input(feature_name, min_value=0.0, value=0.0, format="%.2f", key=f"num_{feature_name}")

# --- 4. SCORING CALCULATOR ---
st.markdown("### 🧮 Integrated Clinical Scores")
with st.expander("SIRS & BISAP Criteria Calculator", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.write("**SIRS Logic**")
        sirs_rr = st.checkbox("Resp Rate > 20/min or PaCO2 < 32 mmHg")
        
        # Auto-calculate SIRS
        s_score = 0
        if user_data.get('Heart rate', 0) > 90: s_score += 1
        if user_data.get('Wbc count', 0) > 12000 or (0 < user_data.get('Wbc count', 5000) < 4000): s_score += 1
        if sirs_rr: s_score += 1
        
        temp_val = str(user_data.get('Temperature', ''))
        if "> 38" in temp_val or "< 36" in temp_val: s_score += 1
        st.info(f"Current SIRS Score: {s_score}")

    with c2:
        st.write("**BISAP Logic**")
        gcs_impaired = st.checkbox("Impaired Mental Status (GCS < 15)")
        
        # Auto-calculate BISAP
        b_score = 0
        if user_data.get('BUN', 0) > 25: b_score += 1
        if gcs_impaired: b_score += 1
        if s_score >= 2: b_score += 1
        if user_data.get('Age', 0) > 60: b_score += 1
        
        pe_val = str(user_data.get('Pleural effusion', '')).lower()
        if "yes" in pe_val or "present" in pe_val: b_score += 1
        st.info(f"Current BISAP Score: {b_score}")

# Sync calculated scores back to the feature list
for f in assets['features']:
    if 'SIRS' in f: user_data[f] = s_score
    if 'BISAP' in f: user_data[f] = b_score

# --- 5. PREDICTION LOGIC ---
st.markdown("---")
if st.button("RUN ANALYSIS", use_container_width=True):
    final_features = []
    
    for col in assets['features']:
        val = user_data.get(col, 0)
        
        # Handle Categorical Encoding
        if col in assets['le_dict']:
            try:
                text_val = str(val).strip()
                if text_val in assets['le_dict'][col].classes_:
                    encoded_val = assets['le_dict'][col].transform([text_val])[0]
                else:
                    encoded_val = 0
                final_features.append(encoded_val)
            except:
                final_features.append(0)
        
        # Handle Numeric Conversion (The Fix for your Error)
        else:
            try:
                if isinstance(val, str):
                    # Strip everything except numbers and decimals
                    clean_num = "".join(c for c in val if c.isdigit() or c == '.')
                    final_features.append(float(clean_num) if clean_num else 0.0)
                else:
                    final_features.append(float(val))
            except (ValueError, TypeError):
                final_features.append(0.0)

    # Execute Prediction
    try:
        final_X = np.array([final_features])
        pred_idx = assets['model'].predict(final_X)[0]
        result = assets['le_target'].classes_[pred_idx]
        
        st.markdown(f'''
        <div class="glass-card" style="text-align: center; border-left: 10px solid #ff4b4b;">
            <h2 style="margin:0;">PREDICTED SEVERITY</h2>
            <h1 style="font-size: 3.5em; color: #ffeb3b !important;">{result.upper()}</h1>
            <p>Based on 32-parameter Random Forest Analysis & Revised Atlanta Classification</p>
        </div>
        ''', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
