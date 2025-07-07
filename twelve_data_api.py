# twelve_data_api.py
import requests
import pandas as pd
from datetime import datetime

API_KEY = "0633d31b59084be59ab4499724fa470c"
BASE_URL = "https://api.twelvedata.com/time_series"

def download(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "interval": interval,
        "start_date": start,
        "end_date": end,
        "apikey": API_KEY,
        "format": "JSON",
        "outputsize": 5000,
        "dp": 4,
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if "values" not in data:
        raise ValueError(f"Ошибка при загрузке данных для {symbol}: {data.get('message', 'Неизвестная ошибка')}")

    df = pd.DataFrame(data["values"])
    df = df.rename(columns={"datetime": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })

    return df.sort_index()