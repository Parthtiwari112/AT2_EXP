# quick_train_from_csv.py
import os, argparse, pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="dataset.csv", help="Path to dataset.csv")
parser.add_argument("--out", default="models", help="Output dir to save models")
args = parser.parse_args()

df = pd.read_csv(args.input, parse_dates=["date"])
# features expected in your dataset.csv
FEATURES = [
    "dayofyear_sin","dayofyear_cos","precip_last_1","precip_last_3","precip_last_7",
    "temp_max_mean_7","temp_min_mean_7","precip_hours_last_7","weathercode"
]
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise SystemExit("Missing features in input CSV: " + ", ".join(missing))

X = df[FEATURES].fillna(0.0)
y_cls = df["will_rain_plus7"].astype(int)
y_reg = df["precip_next_3days"].astype(float)

# Train classifier
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X, y_cls)

# Train regressor
reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
reg.fit(X, y_reg)

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "rain_model.pkl"), "wb") as f:
    pickle.dump(clf, f)
with open(os.path.join(args.out, "precipitation_model.pkl"), "wb") as f:
    pickle.dump(reg, f)

print("Saved models to", args.out)
