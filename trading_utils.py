import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from smart_money_indicators import detect_bos

def prepare_data(df: pd.DataFrame, sequence_length: int = 60):
    try:
        df = df.copy()
        df['BOS'] = detect_bos(df['Close'])

        df.dropna(inplace=True)
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'BOS']

        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df[feature_cols])
        
        X, y = [], []
        for i in range(sequence_length, len(df_scaled)):
            X.append(df_scaled[i - sequence_length:i])
            y.append(df_scaled[i, 3])  # 'Close' цена

        return np.array(X), np.array(y)
    
    except Exception as e:
        raise ValueError(f"Ошибка при подготовке данных: {e}")


def detect_bos(close_series, window=5):
    close_series = close_series.reset_index(drop=True)  # Обеспечим числовой индекс
    bos = [0] * len(close_series)
    for i in range(window, len(close_series) - window):
        prev_highs = close_series[i - window:i]
        next_highs = close_series[i + 1:i + 1 + window]
        current = close_series[i]

        if current > prev_highs.max() and current > next_highs.max():
            bos[i] = 1
    return bos


def detect_order_blocks(df, threshold=0.7):
    ob = pd.Series(0, index=df.index)
    body = abs(df['Open'] - df['Close'])

    mean_body = body.mean()
    for i in range(2, len(df)):
        if body.iloc[i] > mean_body * threshold:
            if df['Close'].iloc[i] > df['Open'].iloc[i]:
                ob.iloc[i] = 1  # Bullish OB
            else:
                ob.iloc[i] = -1  # Bearish OB
    return ob


def detect_fvg(df):
    fvg = pd.Series(0, index=df.index)
    for i in range(2, len(df)):
        if df['Low'].iloc[i] > df['High'].iloc[i - 2]:
            fvg.iloc[i] = 1  # Bullish FVG
        elif df['High'].iloc[i] < df['Low'].iloc[i - 2]:
            fvg.iloc[i] = -1  # Bearish FVG
    return fvg