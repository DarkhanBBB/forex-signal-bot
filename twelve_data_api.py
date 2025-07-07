
import os
import requests
import pandas as pd

API_KEY = os.getenv("TWELVE_DATA_API_KEY")
BASE_URL = "https://api.twelvedata.com/time_series"

def download(symbol: str, interval: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    url = (
        f"{BASE_URL}?symbol={symbol}&interval={interval}"
        f"&start_date={start_date.strftime('%Y-%m-%d %H:%M:%S')}"
        f"&end_date={end_date.strftime('%Y-%m-%d %H:%M:%S')}"
        f"&apikey={API_KEY}&format=JSON&dp=5"
    )
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Ошибка при запросе данных: {response.text}")

    data = response.json()
    if "values" not in data:
        raise ValueError(f"Нет данных в ответе API: {data}")

    df = pd.DataFrame(data["values"])
    df = df.rename(columns={"datetime": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.set_index("Datetime", inplace=True)
    df = df.sort_index()

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0

    return df
