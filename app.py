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
st.markdown("##### *Predictive Analysis Dashboard*")

st.markdown("### 📋 Clinical Parameters")
cols = st.columns(3)

# 100% hardcoded rendering sequence requested by the user
explicit_ui_order = [
    "TEMPERATURE",
    "HEART RATE",
    "RESPIRATORY RATE",
    "BLOOD GLUCOSE LEVEL",
    "WBC",
    "HCT",
    "PLATELET",
    "TB",
    "DB",
    "AST",
    "ALT",
    "BUN",
    "S.CREATININE",
    "LDH",
    "Na",
    "K",
    "Ca",
    "TRIGLYCERIDES",
    "HDL",
    "LDL",
    "SERUM AMYLASE",
    "SERUM LIPASE",
    "CHEST XRAY"
]

user_inputs = {}

# Loop strictly through your requested list to build the UI columns row-by-row
for i, display_name in enumerate(explicit_ui_order):
    with cols[i % 3]:
        if display_name == "TEMPERATURE":
            # Check if model has a custom categorical encoder for Temperature
            matching_key = next((f for f in assets['le_dict'] if "temp" in f.lower()), None)
            choices = assets['le_dict'][matching_key].classes_ if matching_key else ["< 36", "36 - 38", "> 38"]
            user_inputs[display_name] = st.selectbox("Temperature Status", choices)
            
        elif display_name == "CHEST XRAY":
            # Map Chest Xray directly to Pleural Effusion criteria
            matching_key = next((f for f in assets['le_dict'] if "effusion" in f.lower() or "xray" in f.lower()), None)
            choices = assets['le_dict'][matching_key].classes_ if matching_key else ["No", "Yes"]
            user_inputs[display_name] = st.selectbox("Chest X-Ray / Pleural Effusion", choices)
            
        else:
            # All other lab parameters are treated as clean numerical floats
            user_inputs[display_name] = st.number_input(display_name, min_value=0.0, value=0.0, format="%.2f")

# --- 4. PREDICTION ---
if st.button("RUN CLINICAL ANALYSIS", use_container_width=True):
    
    # 1. Calculate clinical scoring values behind the scenes
    sirs_val = 0
    if user_inputs.get('HEART RATE', 0) > 90: sirs_val += 1
    if user_inputs.get('WBC', 0) > 12000 or (0 < user_inputs.get('WBC', 0) < 4000): sirs_val += 1
    
    temp_str = str(user_inputs.get('TEMPERATURE', ''))
    if "> 38" in temp_str or "< 36" in temp_str: sirs_val += 1
    
    bisap_val = 0
    if user_inputs.get('BUN', 0) > 25: bisap_val += 1
    if sirs_val >= 2: bisap_val += 1
    
    # Check age if your model relies on age (defaults to 0 if not captured in your clean list)
    age_feature = next((user_inputs[k] for k in user_inputs if "age" in k.lower()), 0)
    if age_feature > 60: bisap_val += 1
    
    cxr_str = str(user_inputs.get('CHEST XRAY', '')).lower()
    if "yes" in cxr_str or "present" in cxr_str: bisap_val += 1

    # 2. Re-align user inputs back to the exact feature positions expected by the model
    final_features = []
    for col in assets['features']:
        is_calculated = False
        val = 0
        
        # Catch calculated features
        if 'SIRS' in col.upper():
            val = sirs_val
            is_calculated = True
        elif 'BISAP' in col.upper():
            val = bisap_val
            is_calculated = True
        elif any(ex in col.upper() for ex in ['AIP', 'CTSI', 'ALBUMIN', 'CRP']):
            val = 0 # Explicitly zero out dropped/excluded parameters
            is_calculated = True
            
        # Match user inputs against model features regardless of string case variations
        else:
            col_clean = col.upper().replace(" COUNT", "").replace(" STATUS", "").replace(" LEVEL", "").strip()
            if "EFFUSION" in col_clean or "XRAY" in col_clean or "X-RAY" in col_clean:
                val = user_inputs.get("CHEST XRAY", 0)
            elif "GLUCOSE" in col_clean:
                val = user_inputs.get("BLOOD GLUCOSE LEVEL", 0)
            elif "LIPASE" in col_clean:
                val = user_inputs.get("SERUM LIPASE", 0)
            elif "AMYLASE" in col_clean:
                val = user_inputs.get("SERUM AMYLASE", 0)
            elif "CREATININE" in col_clean:
                val = user_inputs.get("S.CREATININE", 0)
            else:
                val = user_inputs.get(col_clean, 0)

        # 3. Handle Label Encoding vs Numbers for the mapped value
        if col in assets['le_dict'] and not is_calculated:
            try:
                encoded = assets['le_dict'][col].transform([str(val).strip()])[0]
                final_features.append(encoded)
            except Exception:
                final_features.append(0)
        else:
            try:
                clean_val = "".join(c for c in str(val) if c.isdigit() or c == '.')
                final_features.append(float(clean_val) if clean_val else 0.0)
            except Exception:
                final_features.append(0.0)

    # 4. Predict Result
    try:
        final_X = np.array([final_features])
        pred_idx = assets['model'].predict(final_X)[0]
        
        if 'le_target' in assets and hasattr(assets['le_target'], 'classes_'):
            result = assets['le_target'].classes_[pred_idx]
        else:
            result = str(pred_idx)
        
        st.markdown(f'''
        <div class="glass-card" style="border-left: 10px solid #ffeb3b;">
            <h2 style="margin:0;">FINAL SEVERITY RESULT</h2>
            <h1 style="color: #ffeb3b !important;">{str(result).upper()}</h1>
            <p style="font-size: 0.9em; opacity: 0.8;">Analysis complete</p>
        </div>
        ''', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
