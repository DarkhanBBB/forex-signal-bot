import pandas as pd
import numpy as np
from twelvedata import TDClient
from trading_utils import prepare_data_for_training

TD_API_KEY = "0633d31b59084be59ab4499724fa470c"
td = TDClient(apikey=TD_API_KEY)

def convert_interval(interval):
    mapping = {
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1day"
    }
    return mapping.get(interval, "1h")

def fetch_data(symbol, interval="1h", outputsize=300):
    try:
        tf = convert_interval(interval)
        data = td.time_series(
            symbol=symbol,
            interval=tf,
            outputsize=outputsize,
            order='ASC'
        ).as_pandas()

        df = data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        df.index = pd.to_datetime(df.index)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    except Exception as e:
        print(f"Ошибка при загрузке данных из Twelve Data: {e}")
        return pd.DataFrame()

# 👉 Добавляем эту функцию:
def get_training_data():
    symbols = ["EUR/USD", "XAU/USD"]
    interval = "1h"
    all_x = []
    all_y = []

    for symbol in symbols:
        df = fetch_data(symbol, interval)
        if not df.empty:
            x, y = prepare_data_for_training(df)
            all_x.append(x)
            all_y.append(y)

    if all_x:
        return np.concatenate(all_x), np.concatenate(all_y)
    else:
        raise ValueError("Нет данных для обучения.")