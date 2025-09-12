"""
retrain.py - example script to retrain models using real historical data.
Replace the data-loading section with the dataset you prepared (Open-Meteo historical API results).
This script trains:
 - RandomForestClassifier -> models/rain_model.pkl
 - RandomForestRegressor  -> models/precipitation_model.pkl
Feature columns expected:
["dayofyear_sin","dayofyear_cos","precip_last_1","precip_last_3","precip_last_7",
 "temp_max_mean_7","temp_min_mean_7","precip_hours_last_7","weathercode"]
Targets:
 - will_rain_plus7 (0/1)
 - precip_next_3days (float)
"""
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load data ---
# Replace this with your processed historical dataset (CSV) that matches expected features.
DATA_PATH = os.path.join(ROOT, "training_data_sample.csv")
print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

FEATURE_COLS = ["dayofyear_sin","dayofyear_cos","precip_last_1","precip_last_3","precip_last_7",
                "temp_max_mean_7","temp_min_mean_7","precip_hours_last_7","weathercode"]

# Basic check
for c in FEATURE_COLS + ["will_rain_plus7","precip_next_3days"]:
    if c not in df.columns:
        raise ValueError(f"Expected column '{c}' not found in {DATA_PATH}")

X = df[FEATURE_COLS].values
y_cls = df["will_rain_plus7"].values
y_reg = df["precip_next_3days"].values

# Train/test split (temporal split recommended in production)
X_train, X_val, y_train_cls, y_val_cls, y_train_reg, y_val_reg = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=42
)

# Train classifier
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train_cls)

# Train regressor
reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
reg.fit(X_train, y_train_reg)

# Evaluation
cls_pred = clf.predict(X_val)
cls_proba = clf.predict_proba(X_val)[:,1] if hasattr(clf, "predict_proba") else None
reg_pred = reg.predict(X_val)

acc = accuracy_score(y_val_cls, cls_pred)
auc = roc_auc_score(y_val_cls, cls_proba) if cls_proba is not None else None
mse = mean_squared_error(y_val_reg, reg_pred)
r2 = r2_score(y_val_reg, reg_pred)

print("Classifier: accuracy=%.4f, auc=%s" % (acc, str(auc)))
print("Regressor: mse=%.4f, r2=%.4f" % (mse, r2))

# Save models
with open(os.path.join(MODEL_DIR, "rain_model.pkl"), "wb") as f:
    pickle.dump(clf, f)
with open(os.path.join(MODEL_DIR, "precipitation_model.pkl"), "wb") as f:
    pickle.dump(reg, f)

print("Models saved to:", MODEL_DIR)
