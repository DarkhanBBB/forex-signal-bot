import numpy as np
import pandas as pd

def prepare_data(df):
    df = df.copy()
    try:
        df["BOS"] = [1 if s['type'] == 'BOS_up' else -1 for s in detect_bos(df)]
        df["FVG"] = [1 if s['type'] == 'FVG_up' else -1 for s in detect_fvg(df)]
        df["OB"] = [1 if s['type'] == 'OB_bullish' else -1 for s in detect_order_blocks(df)]
        df["LQZ"] = [1 if s['type'] == 'equal_highs' else -1 for s in detect_liquidity_zones(df)]

        df.dropna(inplace=True)
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'BOS', 'FVG', 'OB']  # LQZ можно добавить позже
        X = df[features].values
        y = (df['Close'].pct_change().shift(-1) > 0).astype(int)
        y = y[-len(X):]
        return X, y
    except Exception as e:
        raise ValueError(f"Ошибка при расчёте Smart Money индикаторов: {e}")

def confidence_score(model, X_new):
    prediction = model.predict(X_new[-1].reshape(1, -1))[0][0]
    return float(prediction)

def should_enter_trade(confidence, threshold=0.8):
    return confidence > threshold
