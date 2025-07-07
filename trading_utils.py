import pandas as pd


def detect_bos(close_series, window=5):
    bos = [0] * len(close_series)
    for i in range(window, len(close_series) - window):
        prev_highs = close_series[i - window:i]
        next_highs = close_series[i + 1:i + 1 + window]

        current = close_series[i]

        # Убедимся, что сравниваем значения, а не Series
        if (
            current > prev_highs.max() and
            current > next_highs.max()
        ):
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