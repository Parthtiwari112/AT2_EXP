import glob
import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)
files = sorted(glob.glob("data/raw/daily_*.parquet"))

dfs = []
for f in files:
    try:
        d = pd.read_parquet(f)
        dfs.append(d)
    except Exception as e:
        print("Failed to read", f, e)

if not dfs:
    raise SystemExit("No yearly parquet files found in data/raw/")

df = pd.concat(dfs).sort_index()
# remove exact duplicate indices (if any)
df = df[~df.index.duplicated(keep='first')]
out_path = "data/processed/daily_1980_2024.parquet"
df.to_parquet(out_path)
print("Merged ->", out_path, "rows:", len(df))
