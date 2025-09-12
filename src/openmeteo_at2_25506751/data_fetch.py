import requests
import pandas as pd
import time
from typing import List, Optional

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# Default location: Sydney
DEFAULT_LAT = -33.8678
DEFAULT_LON = 151.2073
DEFAULT_TZ = "Australia/Sydney"

def fetch_daily_range(
    start_date: str,
    end_date: str,
    latitude: float = DEFAULT_LAT,
    longitude: float = DEFAULT_LON,
    daily_vars: Optional[List[str]] = None,
    timezone: str = DEFAULT_TZ,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch daily variables from Open-Meteo archive API for the given date range.
    Returns a pandas DataFrame indexed by date (pd.DatetimeIndex).
    """
    if daily_vars is None:
        daily_vars = [
            "precipitation_sum", "rain_sum",
            "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "sunrise", "sunset",
            "windgusts_10m_max", "windspeed_10m_max"
        ]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(daily_vars),
        "timezone": timezone
    }

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=60)
            r.raise_for_status()
            payload = r.json()
            daily = payload.get("daily", {})
            if not daily:
                return pd.DataFrame()
            df = pd.DataFrame(daily)
            if "time" in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')
            else:
                # defensive
                df.index = pd.to_datetime(df.index)
            return df.sort_index()
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(retry_delay)
    return pd.DataFrame()

if __name__ == "__main__":
    import os
    os.makedirs("data/raw", exist_ok=True)
    print("Fetching sample 2010 daily...")
    df2010 = fetch_daily_range("2010-01-01", "2010-12-31")
    print("Rows fetched:", len(df2010))
    if not df2010.empty:
        df2010.to_parquet("data/raw/daily_2010.parquet")
        print("Saved data/raw/daily_2010.parquet")
