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

# Exact desired order mapping dictionary to enforce chronological layout ranking
ORDER_MAPPING = {
    'temperature status': 1,
    'heart rate': 2,
    'respiratory rate': 3,
    'blood glucose level': 4,
    'wbc count': 5,
    'hct': 6,
    'platelet': 7,
    'tb': 8,
    'db': 9,
    'ast': 10,
    'alt': 11,
    'bun': 12,
    's.creatinine': 13,
    'ldh': 14,
    'na': 15,
    'k': 16,
    'ca': 17,
    'triglycerides': 18,
    'hdl': 19,
    'ldl': 20,
    'serum amylase': 21,
    'serum lipase': 22,
    'pleural effusion': 23,
    'chest xray': 23,
    'chest x-ray': 23
}

# Filter out excluded or duplicate model fields completely
excluded_fields = ['SIRS', 'BISAP', 'BISAP Score', 'AIP', 'CTSI', 'SCORE', 'ALBUMIN', 'CRP']
valid_features = [f for f in assets['features'] if not any(ex in f.upper() for ex in excluded_fields)]

# Function to safely rank items based on your precise order list
def get_feature_rank(feature_name):
    fname_lower = feature_name.lower().strip()
    if fname_lower == 'etiology':
        return 0  # Keep demographics/etiology at the absolute top if present
    return ORDER_MAPPING.get(fname_lower, 999) # Append unanticipated fields to the bottom

# Sort features strictly by your layout order rank
input_features = sorted(valid_features, key=get_feature_rank)

user_data = {}
st.markdown("### 📋 Clinical Parameters")
cols = st.columns(3)

for i, feature_name in enumerate(input_features):
    with cols[i % 3]:
        # Handle Etiology Selectbox
        if feature_name == 'Etiology':
            all_etiologies = assets['le_dict']['Etiology'].classes_
            filtered_etiologies = [e for e in all_etiologies if e not in ['AIP', 'CTSI']]
            user_data[feature_name] = st.selectbox("Etiology", filtered_etiologies)
            
        # Handle Pleural Effusion / Chest X-Ray presentation cleanly
        elif feature_name.lower() in ['pleural effusion', 'chest xray', 'chest x-ray']:
            choices = assets['le_dict'][feature_name].classes_ if feature_name in assets['le_dict'] else ["No", "Yes"]
            user_data[feature_name] = st.selectbox("Chest X-Ray / Pleural Effusion", choices)
            
        # Handle Categorical Fields
        elif feature_name in assets['le_dict']:
            user_data[feature_name] = st.selectbox(feature_name, assets['le_dict'][feature_name].classes_)
            
        # Handle Duration Fields
        elif "duration" in feature_name.lower():
            choice = st.selectbox(feature_name, ["Lesser than 3 days", "Greater than 3 days"])
            user_data[feature_name] = "1- 3 days" if "Lesser" in choice else "> 3 days"
            
        # Handle Numeric Fields
        else:
            user_data[feature_name] = st.number_input(feature_name, min_value=0.0, value=0.0, format="%.2f")

# --- 4. PREDICTION ---
if st.button("RUN CLINICAL ANALYSIS", use_container_width=True):
    # Dynamic verification targeting critical features
    critical_vitals = ['Age', 'Heart rate', 'Wbc count', 'SBP', 'BMI']
    vitals_to_check = [v for v in critical_vitals if any(f.lower() == v.lower() for f in assets['features'])]
    missing_data = [v for v in vitals_to_check if user_data.get(v, 0) == 0]
    
    if missing_data:
        st.warning(f"⚠️ **Incomplete Data:** Please provide valid values for {', '.join(missing_data)}.")
    else:
        # Backend Calculations (SIRS/BISAP)
        sirs_val = 0
        if user_data.get('Heart rate', 0) > 90: sirs_val += 1
        if user_data.get('Wbc count', 0) > 12000 or (0 < user_data.get('Wbc count', 0) < 4000): sirs_val += 1
        
        temp = str(user_data.get('Temperature Status', ''))
        if "> 38" in temp or "< 36" in temp: sirs_val += 1
        
        bisap_val = 0
        if user_data.get('BUN', 0) > 25: bisap_val += 1
        if sirs_val >= 2: bisap_val += 1
        if user_data.get('Age', 0) > 60: bisap_val += 1
        
        # Intercept chest imaging checks cleanly
        pe_val = ""
        for k, v in user_data.items():
            if "effusion" in k.lower() or "xray" in k.lower() or "x-ray" in k.lower():
                pe_val = str(v).lower()
                break
        if "yes" in pe_val or "present" in pe_val: bisap_val += 1

        # Final Feature Vector Generation
        final_features = []
        for col in assets['features']:
            is_calculated = False
            
            if 'SIRS' in col.upper(): 
                val = sirs_val
                is_calculated = True
            elif 'BISAP' in col.upper(): 
                val = bisap_val
                is_calculated = True
            elif any(ex in col.upper() for ex in ['AIP', 'CTSI', 'ALBUMIN', 'CRP']): 
                val = 0 
                is_calculated = True
            else: 
                val = user_data.get(col, 0)

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

        # Predict Result
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
            st.error(f"Analysis failed: {e}")import streamlit as st
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

# Exact desired order mapping dictionary to enforce chronological layout ranking
ORDER_MAPPING = {
    'temperature status': 1,
    'heart rate': 2,
    'respiratory rate': 3,
    'blood glucose level': 4,
    'wbc count': 5,
    'hct': 6,
    'platelet': 7,
    'tb': 8,
    'db': 9,
    'ast': 10,
    'alt': 11,
    'bun': 12,
    's.creatinine': 13,
    'ldh': 14,
    'na': 15,
    'k': 16,
    'ca': 17,
    'triglycerides': 18,
    'hdl': 19,
    'ldl': 20,
    'serum amylase': 21,
    'serum lipase': 22,
    'pleural effusion': 23,
    'chest xray': 23,
    'chest x-ray': 23
}

# Filter out excluded or duplicate model fields completely
excluded_fields = ['SIRS', 'BISAP', 'BISAP Score', 'AIP', 'CTSI', 'SCORE', 'ALBUMIN', 'CRP']
valid_features = [f for f in assets['features'] if not any(ex in f.upper() for ex in excluded_fields)]

# Function to safely rank items based on your precise order list
def get_feature_rank(feature_name):
    fname_lower = feature_name.lower().strip()
    if fname_lower == 'etiology':
        return 0  # Keep demographics/etiology at the absolute top if present
    return ORDER_MAPPING.get(fname_lower, 999) # Append unanticipated fields to the bottom

# Sort features strictly by your layout order rank
input_features = sorted(valid_features, key=get_feature_rank)

user_data = {}
st.markdown("### 📋 Clinical Parameters")
cols = st.columns(3)

for i, feature_name in enumerate(input_features):
    with cols[i % 3]:
        # Handle Etiology Selectbox
        if feature_name == 'Etiology':
            all_etiologies = assets['le_dict']['Etiology'].classes_
            filtered_etiologies = [e for e in all_etiologies if e not in ['AIP', 'CTSI']]
            user_data[feature_name] = st.selectbox("Etiology", filtered_etiologies)
            
        # Handle Pleural Effusion / Chest X-Ray presentation cleanly
        elif feature_name.lower() in ['pleural effusion', 'chest xray', 'chest x-ray']:
            choices = assets['le_dict'][feature_name].classes_ if feature_name in assets['le_dict'] else ["No", "Yes"]
            user_data[feature_name] = st.selectbox("Chest X-Ray / Pleural Effusion", choices)
            
        # Handle Categorical Fields
        elif feature_name in assets['le_dict']:
            user_data[feature_name] = st.selectbox(feature_name, assets['le_dict'][feature_name].classes_)
            
        # Handle Duration Fields
        elif "duration" in feature_name.lower():
            choice = st.selectbox(feature_name, ["Lesser than 3 days", "Greater than 3 days"])
            user_data[feature_name] = "1- 3 days" if "Lesser" in choice else "> 3 days"
            
        # Handle Numeric Fields
        else:
            user_data[feature_name] = st.number_input(feature_name, min_value=0.0, value=0.0, format="%.2f")

# --- 4. PREDICTION ---
if st.button("RUN CLINICAL ANALYSIS", use_container_width=True):
    # Dynamic verification targeting critical features
    critical_vitals = ['Age', 'Heart rate', 'Wbc count', 'SBP', 'BMI']
    vitals_to_check = [v for v in critical_vitals if any(f.lower() == v.lower() for f in assets['features'])]
    missing_data = [v for v in vitals_to_check if user_data.get(v, 0) == 0]
    
    if missing_data:
        st.warning(f"⚠️ **Incomplete Data:** Please provide valid values for {', '.join(missing_data)}.")
    else:
        # Backend Calculations (SIRS/BISAP)
        sirs_val = 0
        if user_data.get('Heart rate', 0) > 90: sirs_val += 1
        if user_data.get('Wbc count', 0) > 12000 or (0 < user_data.get('Wbc count', 0) < 4000): sirs_val += 1
        
        temp = str(user_data.get('Temperature Status', ''))
        if "> 38" in temp or "< 36" in temp: sirs_val += 1
        
        bisap_val = 0
        if user_data.get('BUN', 0) > 25: bisap_val += 1
        if sirs_val >= 2: bisap_val += 1
        if user_data.get('Age', 0) > 60: bisap_val += 1
        
        # Intercept chest imaging checks cleanly
        pe_val = ""
        for k, v in user_data.items():
            if "effusion" in k.lower() or "xray" in k.lower() or "x-ray" in k.lower():
                pe_val = str(v).lower()
                break
        if "yes" in pe_val or "present" in pe_val: bisap_val += 1

        # Final Feature Vector Generation
        final_features = []
        for col in assets['features']:
            is_calculated = False
            
            if 'SIRS' in col.upper(): 
                val = sirs_val
                is_calculated = True
            elif 'BISAP' in col.upper(): 
                val = bisap_val
                is_calculated = True
            elif any(ex in col.upper() for ex in ['AIP', 'CTSI', 'ALBUMIN', 'CRP']): 
                val = 0 
                is_calculated = True
            else: 
                val = user_data.get(col, 0)

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

        # Predict Result
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
