import pandas as pd


def detect_bos(close_series, window=5):
    bos = [0] * len(close_series)
    for i in range(window, len(close_series) - window):
        prev_highs = close_series[i - window:i]
        next_highs = close_series[i + 1:i + 1 + window]
        if close_series.iloc[i] > prev_highs.max() and close_series.iloc[i] > next_highs.max():
            bos[i] = 1
    return bos


def detect_order_blocks(df, threshold=0.7):
    blocks = []
    body = abs(df['Open'] - df['Close'])
    for i in range(2, len(df)):
        if body[i] > body[:i].mean() * threshold:
            block_type = 'Bullish OB' if df['Close'][i] > df['Open'][i] else 'Bearish OB'
            blocks.append((i, block_type))
    return blocks


def detect_fvg(df):
    fvg_list = []
    for i in range(2, len(df)):
        if df['Low'][i] > df['High'][i - 2]:
            fvg_list.append((i, 'Bullish FVG'))
        elif df['High'][i] < df['Low'][i - 2]:
            fvg_list.append((i, 'Bearish FVG'))
    return fvg_list
