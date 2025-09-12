import os
import sys
import pandas as pd

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.openmeteo_at2_25506751.features import (
    build_features_for_date,
    make_label_classification,
    make_label_regression,
)


def build_features_table(
    daily_path="data/processed/daily_1980_2024.parquet",
    out_path="data/processed/features_table.parquet",
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.read_parquet(daily_path)
    df = df.sort_index()
    rows = []

    # last input date must allow label for +7 days
    last_input_date = df.index.max() - pd.Timedelta(days=7)
    start_date = df.index.min()

    total_days = (last_input_date - start_date).days + 1
    print("Building features for", total_days, "dates.")

    cur = start_date
    count = 0
    while cur <= last_input_date:
        feats = build_features_for_date(df, cur)
        if feats is not None:
            lab_c = make_label_classification(df, cur)
            lab_r = make_label_regression(df, cur)
            if lab_c is not None and lab_r is not None:
                r = feats.to_dict()
                r["label_class"] = lab_c # type: ignore
                r["label_reg"] = lab_r # type: ignore
                r["input_date"] = cur
                rows.append(r)
        cur = cur + pd.Timedelta(days=1)
        count += 1
        if count % 365 == 0:
            print("Processed", count, "days...")

    features_table = pd.DataFrame(rows).set_index("input_date").sort_index()
    features_table.to_parquet(out_path)
    print("Saved features_table:", out_path, "rows:", len(features_table))


if __name__ == "__main__":
    build_features_table()
