import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

def prepare_data(df, window_size=50):
    df = df.copy()
    scaler = MinMaxScaler()
    df[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(
        df[['open', 'high', 'low', 'close', 'volume']]
    )

    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(df[['open', 'high', 'low', 'close', 'volume']].iloc[i - window_size:i].values)
        y.append(1 if df['close'].iloc[i] > df['close'].iloc[i - 1] else 0)

    return np.array(X), np.array(y), scaler

# ✅ Функция для обучения модели (без scaler)
def prepare_data_for_training(df, window_size=50):
    df = df.copy()
    scaler = MinMaxScaler()
    df[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(
        df[['open', 'high', 'low', 'close', 'volume']]
    )

    X, y = [], []
    for i in range(window_size, len(df)):
        X.append(df[['open', 'high', 'low', 'close', 'volume']].iloc[i - window_size:i].values)
        y.append(1 if df['close'].iloc[i] > df['close'].iloc[i - 1] else 0)

    return np.array(X), np.array(y)

def confidence_score(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_class = (y_pred > 0.5).astype(int).flatten()
    return accuracy_score(y_test, y_pred_class)

def should_enter_trade(prediction, threshold=0.8):
    return prediction > threshold
