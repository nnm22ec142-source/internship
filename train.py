import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# 1. Load the Excel file
excel_file = 'Acute_pancreatitis_RF_imputed.xlsx'
print(f"Reading {excel_file}...")
df = pd.read_excel(excel_file, sheet_name=1)

# 2. Identify the Target (Severity)
target_col = 'Severity of pancreatitis as per Atlanta'

# Clean the target column (remove extra spaces and handle missing values)
df[target_col] = df[target_col].astype(str).str.strip()
df = df[df[target_col] != 'nan'] # Drop rows where target is missing

# 3. Clean Data: Remove Dates and IDs
cols_to_drop = [target_col, 'Ip number', 'Timestamp', 'IP NO', 'SIRS \nA patient meets SIRS...', 'Severity as per bisap...']
X = df.drop(columns=[c for c in df.columns if any(x in c for x in cols_to_drop)], errors='ignore')

# 4. Clean Numeric Columns (Handle "86bpm" or "120/80")
for col in X.columns:
    if X[col].dtype == 'object':
        # Try to see if it should be numeric
        # This removes "bpm", "mmHg", etc.
        converted = pd.to_numeric(X[col].astype(str).str.extract('(\d+)', expand=False), errors='coerce')
        if converted.notnull().sum() > len(X) * 0.8: # If 80% are numbers
            X[col] = converted

# 5. Dynamic Encoding
le_dict = {}
# Find text columns
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    # Clean spaces from categories
    X[col] = X[col].astype(str).str.strip()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# Encode the Target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(df[target_col])

# Fill any remaining NaNs with 0
X = X.fillna(0)

# 6. Train & Save
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_encoded)

assets = {
    'model': model,
    'le_dict': le_dict,
    'le_target': le_target,
    'features': X.columns.tolist()
}

with open('model.pkl', 'wb') as f:
    pickle.dump(assets, f)

print(f"✅ Success! Trained on {len(X.columns)} features.")
print(f"Categories found: {list(le_dict.keys())}")