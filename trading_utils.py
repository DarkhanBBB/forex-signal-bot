import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_data(df, sequence_length=50):
    data = df[['open', 'high', 'low', 'close', 'volume']].values
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i])
        y.append(data[i, 3])  # close

    X, y = np.array(X), np.array(y)
    return X, y, scaler


def confidence_score(predicted_price, current_price):
    change = abs(predicted_price - current_price) / current_price
    score = 1.0 - change
    return round(float(score), 4)


def should_enter_trade(confidence, threshold=0.80):
    return confidence >= threshold
