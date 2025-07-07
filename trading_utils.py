import numpy as np
import pandas as pd

def prepare_data(df, window_size=10):
    df = df.copy()
    df.dropna(inplace=True)
    df["target"] = df["close"].shift(-1) > df["close"]
    df.dropna(inplace=True)

    features = ["open", "high", "low", "close"]
    x = []
    y = []
    for i in range(window_size, len(df)):
        x.append(df[features].iloc[i - window_size:i].values)
        y.append(int(df["target"].iloc[i]))

    return np.array(x), np.array(y), df.iloc[window_size:]

def confidence_score(prob):
    return abs(prob - 0.5) * 2

def should_enter_trade(prob, threshold=0.8):
    return confidence_score(prob) >= threshold
