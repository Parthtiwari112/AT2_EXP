import pandas as pd
import numpy as np

def build_features_for_date(daily_df: pd.DataFrame, input_date: pd.Timestamp):
    """
    Build a feature vector (pd.Series) for a given input_date using only data up to input_date.
    daily_df must be indexed by date (pd.DatetimeIndex) and sorted ascending.
    """
    if not isinstance(input_date, pd.Timestamp):
        input_date = pd.Timestamp(input_date)

    df = daily_df.loc[:input_date].copy().sort_index()
    if df.empty:
        return None

    features = {}

    # choose precipitation column name if different
    precip_col = None
    for c in ["precipitation_sum", "rain_sum"]:
        if c in df.columns:
            precip_col = c
            break
    if precip_col is None:
        raise KeyError("No precipitation column found in daily_df")

    # lag features: mean & sum over last N days
    for days in [1, 3, 7, 14]:
        window = df[precip_col].iloc[-days:] if len(df) >= days else df[precip_col]
        features[f"precip_mean_last_{days}d"] = float(window.mean()) if len(window) > 0 else 0.0
        features[f"precip_sum_last_{days}d"] = float(window.sum()) if len(window) > 0 else 0.0

    # temperature aggregates if available
    if "temperature_2m_mean" in df.columns:
        window = df["temperature_2m_mean"].iloc[-7:] if len(df) >= 7 else df["temperature_2m_mean"]
        features["temp_mean_last_7d"] = float(window.mean()) if len(window) > 0 else np.nan
        features["temp_std_last_7d"] = float(window.std()) if len(window) > 0 else np.nan
    else:
        features["temp_mean_last_7d"] = np.nan
        features["temp_std_last_7d"] = np.nan

    # day-of-year cyclical features
    doy = input_date.timetuple().tm_yday
    features["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # yesterday rain indicator
    if len(df) >= 1:
        yesterday = df[precip_col].iloc[-1]
        features["yesterday_rain"] = float(yesterday > 0)
    else:
        features["yesterday_rain"] = 0.0

    return pd.Series(features, name=pd.to_datetime(input_date))


def make_label_classification(daily_df: pd.DataFrame, input_date: pd.Timestamp):
    """
    Binary label: will there be rainfall (>0 mm) on input_date + 7?
    Returns 1, 0, or None (if label can't be built).
    """
    if not isinstance(input_date, pd.Timestamp):
        input_date = pd.Timestamp(input_date)

    target = input_date + pd.Timedelta(days=7)
    try:
        precip_col = "precipitation_sum" if "precipitation_sum" in daily_df.columns else "rain_sum"
        val = daily_df.loc[target, precip_col]
        if pd.isna(val):
            return None
        return int(val > 0)
    except Exception:
        return None


def make_label_regression(daily_df: pd.DataFrame, input_date: pd.Timestamp):
    """
    Regression label: cumulative precipitation in input_date+1 .. input_date+3 (mm).
    """
    if not isinstance(input_date, pd.Timestamp):
        input_date = pd.Timestamp(input_date)
    start = input_date + pd.Timedelta(days=1)
    end = input_date + pd.Timedelta(days=3)
    precip_col = "precipitation_sum" if "precipitation_sum" in daily_df.columns else "rain_sum"
    try:
        s = daily_df.loc[start:end, precip_col].sum(min_count=1)
        if pd.isna(s):
            return None
        return float(s)
    except Exception:
        return None
