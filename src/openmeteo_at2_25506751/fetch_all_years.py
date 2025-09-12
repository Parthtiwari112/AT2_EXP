import os
import sys

# Ensure the project root is in sys.path so "src" can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.openmeteo_at2_25506751.data_fetch import fetch_daily_range

os.makedirs("data/raw", exist_ok=True)

start_year = 1980
end_year = 2024

for year in range(start_year, end_year + 1):
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    out_path = f"data/raw/daily_{year}.parquet"
    if os.path.exists(out_path):
        print(f"{out_path} already exists, skipping.")
        continue
    print(f"Fetching year {year} ...")
    df = fetch_daily_range(start, end)
    if df is None or df.empty:
        print(f"No data for {year}, skipped.")
    else:
        df.to_parquet(out_path)
        print(f"Saved {out_path}, rows {len(df)}")
