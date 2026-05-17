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

# 1. Primary parameters locked at the absolute start
primary_parameters = [
    "Age",
    "Sex",
    "Duration of symptoms",
    "Etiology",
    "BMI"
]

# 2. Your strict laboratory/vital sequence
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

# Create a normalized lookup map of all manual layout fields for background matching
normalized_explicit = {
    "age": "Age", "sex": "Sex", "gender": "Sex", "duration": "Duration of symptoms", "etiology": "Etiology", "bmi": "BMI",
    "temp": "TEMPERATURE", "heart": "HEART RATE", "respiratory": "RESPIRATORY RATE", 
    "glucose": "BLOOD GLUCOSE LEVEL", "wbc": "WBC", "hct": "HCT", "platelet": "PLATELET", 
    "tb": "TB", "db": "DB", "ast": "AST", "alt": "ALT", "bun": "BUN", 
    "creatinine": "S.CREATININE", "ldh": "LDH", "na": "Na", "k": "K", "ca": "Ca", 
    "triglyceride": "TRIGLYCERIDES", "hdl": "HDL", "ldl": "LDL", "amylase": "SERUM AMYLASE", 
    "lipase": "SERUM LIPASE", "effusion": "CHEST XRAY", "xray": "CHEST XRAY", "x-ray": "CHEST XRAY"
}

# 3. Identify any remaining backend parameters that aren't accounted for yet
excluded_fields = ['SIRS', 'BISAP', 'BISAP Score', 'AIP', 'CTSI', 'SCORE', 'ALBUMIN', 'CRP', 'BMI 2']
remaining_backend_features = []

for raw_feat in assets['features']:
    feat_upper = raw_feat.upper()
    if any(ex in feat_upper for ex in excluded_fields):
        continue
    matched_explicit = any(k in raw_feat.lower() for k in normalized_explicit.keys())
    if not matched_explicit:
        remaining_backend_features.append(raw_feat)

# 4. Construct complete sequential render list
final_ui_render_list = primary_parameters + explicit_ui_order + remaining_backend_features

user_inputs = {}

# Render fields strictly in order
for i, display_name in enumerate(final_ui_render_list):
    with cols[i % 3]:
        # Handle Primary Top Parameters
        if display_name == "Age":
            user_inputs[display_name] = st.number_input("Age", min_value=0.0, max_value=120.0, value=0.0, format="%.2f")
            
        elif display_name == "Sex":
            matching_key = next((f for f in assets['le_dict'] if "sex" in f.lower() or "gender" in f.lower()), None)
            choices = assets['le_dict'][matching_key].classes_ if matching_key else ["Male", "Female"]
            user_inputs[display_name] = st.selectbox("Sex", choices)
            
        elif display_name == "Duration of symptoms":
            matching_key = next((f for f in assets['le_dict'] if "duration" in f.lower() or "symptom" in f.lower()), None)
            choices = assets['le_dict'][matching_key].classes_ if matching_key else ["Lesser than 3 days", "Greater than 3 days"]
            user_inputs[display_name] = st.selectbox("Duration of Symptoms", choices)
            
        elif display_name == "Etiology":
            all_etiologies = assets['le_dict']['Etiology'].classes_ if 'Etiology' in assets['le_dict'] else ["Biliary", "Alcoholic", "Idiopathic"]
            filtered_etiologies = [e for e in all_etiologies if e not in ['AIP', 'CTSI']]
            user_inputs[display_name] = st.selectbox("Etiology", filtered_etiologies)
            
        elif display_name == "BMI":
            user_inputs[display_name] = st.number_input("BMI", min_value=0.0, value=0.0, format="%.2f")
            
        # Handle Explicit Vitals Sequence
        elif display_name == "TEMPERATURE":
            matching_key = next((f for f in assets['le_dict'] if "temp" in f.lower()), None)
            choices = assets['le_dict'][matching_key].classes_ if matching_key else ["< 36", "36 - 38", "> 38"]
            user_inputs[display_name] = st.selectbox("Temperature Status", choices)
            
        elif display_name == "CHEST XRAY":
            matching_key = next((f for f in assets['le_dict'] if "effusion" in f.lower() or "xray" in f.lower()), None)
            choices = assets['le_dict'][matching_key].classes_ if matching_key else ["No", "Yes"]
            user_inputs[display_name] = st.selectbox("Chest X-Ray / Pleural Effusion", choices)
            
        # Handle Remaining Appended Features Dynamically
        elif display_name in remaining_backend_features:
            if display_name in assets['le_dict']:
                user_inputs[display_name] = st.selectbox(display_name, assets['le_dict'][display_name].classes_)
            else:
                user_inputs[display_name] = st.number_input(display_name, min_value=0.0, value=0.0, format="%.2f")
                
        # Handle Standard Numeric Explicit Inputs
        else:
            user_inputs[display_name] = st.number_input(display_name, min_value=0.0, value=0.0, format="%.2f")

# --- 4. PREDICTION ---
if st.button("RUN CLINICAL ANALYSIS", use_container_width=True):
    
    # Extract clinical calculation metrics
    sirs_val = 0
    if user_inputs.get('HEART RATE', 0) > 90: sirs_val += 1
    if user_inputs.get('WBC', 0) > 12000 or (0 < user_inputs.get('WBC', 0) < 4000): sirs_val += 1
    
    temp_str = str(user_inputs.get('TEMPERATURE', ''))
    if "> 38" in temp_str or "< 36" in temp_str: sirs_val += 1
    
    bisap_val = 0
    if user_inputs.get('BUN', 0) > 25: bisap_val += 1
    if sirs_val >= 2: bisap_val += 1
    if user_inputs.get('Age', 0) > 60: bisap_val += 1
    
    cxr_str = str(user_inputs.get('CHEST XRAY', '')).lower()
    if "yes" in cxr_str or "present" in cxr_str: bisap_val += 1

    # Align UI mapped inputs directly to model features vector matrix
    final_features = []
    for col in assets['features']:
        is_calculated = False
        val = 0
        
        if 'SIRS' in col.upper():
            val = sirs_val
            is_calculated = True
        elif 'BISAP' in col.upper():
            val = bisap_val
            is_calculated = True
        elif any(ex in col.upper() for ex in ['AIP', 'CTSI', 'ALBUMIN', 'CRP', 'BMI 2']):
            val = 0 
            is_calculated = True
        else:
            col_clean = col.upper().replace(" COUNT", "").replace(" STATUS", "").replace(" LEVEL", "").strip()
            
            # Map clean strings back to their UI variables
            if col.lower() in ['age', 'sex', 'gender', 'etiology', 'bmi']:
                if col.lower() == 'age': val = user_inputs.get("Age", 0)
                elif col.lower() in ['sex', 'gender']: val = user_inputs.get("Sex", 0)
                elif col.lower() == 'bmi': val = user_inputs.get("BMI", 0)
                else: val = user_inputs.get("Etiology", 0)
            elif "duration" in col.lower() or "symptom" in col.lower():
                val = user_inputs.get("Duration of symptoms", 0)
            elif "EFFUSION" in col_clean or "XRAY" in col_clean or "X-RAY" in col_clean:
                val = user_inputs.get("CHEST XRAY", 0)
            elif "GLUCOSE" in col_clean:
                val = user_inputs.get("BLOOD GLUCOSE LEVEL", 0)
            elif "LIPASE" in col_clean:
                val = user_inputs.get("SERUM LIPASE", 0)
            elif "AMYLASE" in col_clean:
                val = user_inputs.get("SERUM AMYLASE", 0)
            elif "CREATININE" in col_clean:
                val = user_inputs.get("S.CREATININE", 0)
            elif col in user_inputs:
                val = user_inputs.get(col, 0)
            else:
                val = user_inputs.get(col_clean, 0)

        # Vector processing transformations
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

    # Output prediction
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
