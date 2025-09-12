import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

def train_and_save(features_path="data/processed/features_table.parquet", out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = pd.read_parquet(features_path)
    df = df.sort_index()

    # select features automatically
    feature_cols = [c for c in df.columns if c.startswith(("precip_","temp_","doy_","yesterday_"))]
    print("Using feature columns:", feature_cols)

    # drop rows with missing labels
    df = df.dropna(subset=["label_class", "label_reg"])

    # splits (by date)
    train = df.loc[:'2021-12-31']
    val = df.loc['2022-01-01':'2023-12-31']
    test = df.loc['2024-01-01':'2024-12-31']

    X_train, y_train_clf = train[feature_cols], train["label_class"]
    X_val, y_val_clf = val[feature_cols], val["label_class"]
    X_test, y_test_clf = test[feature_cols], test["label_class"]

    X_train_reg, y_train_reg = train[feature_cols], train["label_reg"]
    X_val_reg, y_val_reg = val[feature_cols], val["label_reg"]
    X_test_reg, y_test_reg = test[feature_cols], test["label_reg"]

    # classification pipeline
    clf_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    clf_pipe.fit(X_train, y_train_clf)
    # evaluate on validation and test
    yhat_val = clf_pipe.predict(X_val)
    yhat_proba_val = clf_pipe.predict_proba(X_val)[:,1] if hasattr(clf_pipe, "predict_proba") else None
    report_val = classification_report(y_val_clf, yhat_val, output_dict=True)
    roc_val = roc_auc_score(y_val_clf, yhat_proba_val) if yhat_proba_val is not None else None

    yhat_test = clf_pipe.predict(X_test)
    yhat_proba_test = clf_pipe.predict_proba(X_test)[:,1] if hasattr(clf_pipe, "predict_proba") else None
    report_test = classification_report(y_test_clf, yhat_test, output_dict=True)
    roc_test = roc_auc_score(y_test_clf, yhat_proba_test) if yhat_proba_test is not None else None

    # save classifier
    clf_path = os.path.join(out_dir, "rain_class_baseline.joblib")
    joblib.dump(clf_pipe, clf_path)
    print("Saved classifier ->", clf_path)

    # regression pipeline
    reg_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    reg_pipe.fit(X_train_reg, y_train_reg)

    yhat_reg_val = reg_pipe.predict(X_val_reg)
    val_mae = mean_absolute_error(y_val_reg, yhat_reg_val)
    val_rmse = sqrt(mean_squared_error(y_val_reg, yhat_reg_val))
    val_r2 = r2_score(y_val_reg, yhat_reg_val)

    yhat_reg_test = reg_pipe.predict(X_test_reg)
    test_mae = mean_absolute_error(y_test_reg, yhat_reg_test)
    test_rmse = sqrt(mean_squared_error(y_test_reg, yhat_reg_test))
    test_r2 = r2_score(y_test_reg, yhat_reg_test)

    reg_path = os.path.join(out_dir, "precip_reg_baseline.joblib")
    joblib.dump(reg_pipe, reg_path)
    print("Saved regressor ->", reg_path)

    # Save metadata: feature columns
    meta = {"feature_columns": feature_cols}
    with open(os.path.join(out_dir, "feature_columns.json"), "w") as f:
        json.dump(meta, f)

    # Save reports/metrics
    metrics = {
        "classification": {
            "val": {"report": report_val, "roc_auc": roc_val},
            "test": {"report": report_test, "roc_auc": roc_test},
        },
        "regression": {
            "val": {"mae": val_mae, "rmse": val_rmse, "r2": val_r2},
            "test": {"mae": test_mae, "rmse": test_rmse, "r2": test_r2},
        }
    }
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved reports/metrics.json")

if __name__ == "__main__":
    train_and_save()
