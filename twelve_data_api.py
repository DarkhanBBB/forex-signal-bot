import pandas as pd
from twelvedata import TDClient

TD_API_KEY = "0633d31b59084be59ab4499724fa470c"
td = TDClient(apikey=TD_API_KEY)

# Преобразование таймфреймов
def convert_interval(interval):
    mapping = {
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1day"
    }
    return mapping.get(interval, "1h")

def fetch_data(symbol, interval="1h", outputsize=120):
    try:
        tf = convert_interval(interval)
        data = td.time_series(
            symbol=symbol,
            interval=tf,
            outputsize=outputsize,
            order='ASC'
        ).as_pandas()

        # Преобразуем в нужный формат
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
