# --------------------------------------------------------------
#  data.py  –  fetch data from open meteo
# --------------------------------------------------------------
import os
import requests
import numpy as np
import pandas as pd

from config import LAT, LON, CACHE_CSV, TRAIN_YEARS, PREDICT_YEAR, SEQ_LEN


def fetch_open_meteo(start: str, end: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   LAT,
        "longitude":  LON,
        "start_date": start,
        "end_date":   end,
        "daily":      "temperature_2m_mean",
        "timezone":   "Europe/Athens",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    raw = r.json()["daily"]
    df = pd.DataFrame({"date": raw["time"], "temp": raw["temperature_2m_mean"]})
    df["date"] = pd.to_datetime(df["date"])
    df["temp"] = df["temp"].interpolate()   #fill any rare nan gaps
    return df


def load_daily() -> pd.DataFrame:
    #return a daily DataFrame. 
    #uses the CSV cache if it fully covers the required date range
    #downloads and saves otherwise.
    
    all_years = sorted(set(TRAIN_YEARS + [PREDICT_YEAR]))
    start = f"{min(all_years)}-01-01"
    end = f"{max(all_years)}-12-31"

    if CACHE_CSV and os.path.exists(CACHE_CSV):
        cached = pd.read_csv(CACHE_CSV, parse_dates=["date"])
        cached_start = cached["date"].min().strftime("%Y-%m-%d")
        cached_end = cached["date"].max().strftime("%Y-%m-%d")

        if cached_start <= start and cached_end >= end:
            print(f"loaded from cache: {CACHE_CSV}")
            return cached

        print(f"cache only covers {cached_start} --> {cached_end}. Re-fetching …")

    print(f"fetching Open-Meteo data {start} --> {end} …")
    df = fetch_open_meteo(start, end)

    if CACHE_CSV:
        df.to_csv(CACHE_CSV, index=False)
        print(f"saved to cache: {CACHE_CSV}")

    return df


def build_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    #Aggregate daily rows into monthly averages. Returns [year, month, temp]
    daily = daily.copy()
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    monthly = (
        daily.groupby(["year", "month"])["temp"]
             .mean()
             .reset_index()
             .sort_values(["year", "month"])
             .reset_index(drop=True)
    )
    return monthly


def prepare_training_data(monthly: pd.DataFrame):
    """
    Normalise using training-year statistics and build sliding-window sequences.

    Returns:
        X          – (N, SEQ_LEN) input sequences  [normalised]
        y          – (N,)         targets           [normalised]
        norm_train – full normalised training series (used as inference seed)
        mu, sigma  – scalars for de-normalising predictions
    """
    train_temps = (
        monthly[monthly["year"].isin(TRAIN_YEARS)]["temp"]
        .values.astype(np.float32)
    )
    mu = train_temps.mean()
    sigma = train_temps.std()
    norm_train = (train_temps - mu) / sigma

    X, y = [], []
    for i in range(len(norm_train) - SEQ_LEN):
        X.append(norm_train[i : i + SEQ_LEN])
        y.append(norm_train[i + SEQ_LEN])

    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.float32),norm_train, float(mu), float(sigma) )