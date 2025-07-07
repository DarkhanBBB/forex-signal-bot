import os
import pandas as pd
import requests

API_KEY = os.getenv("TWELVE_DATA_API_KEY") or "0633d31b59084be59ab4499724fa470c"

def download(symbol, interval, start, end):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
        "apikey": API_KEY,
        "format": "JSON",
        "outputsize": 5000,
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "values" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
