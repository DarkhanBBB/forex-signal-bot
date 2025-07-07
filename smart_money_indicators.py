import pandas as pd

def detect_bos(df: pd.DataFrame):
    """
    Пример простого индикатора Break of Structure (BOS)
    """
    bos_signals = []

    for i in range(2, len(df)):
        if df["high"].iloc[i - 1] > df["high"].iloc[i - 2] and df["high"].iloc[i] < df["high"].iloc[i - 1]:
            bos_signals.append("BOS down")
        elif df["low"].iloc[i - 1] < df["low"].iloc[i - 2] and df["low"].iloc[i] > df["low"].iloc[i - 1]:
            bos_signals.append("BOS up")
        else:
            bos_signals.append(None)

    # Подгоняем длину
    bos_signals = [None, None] + bos_signals
    df["bos"] = bos_signals

    return df


# ======= Fair Value Gap (FVG) =======
def detect_fvg(df: pd.DataFrame):
    fvg_zones = []
    for i in range(2, len(df)):
        low_prev = df['Low'].iloc[i - 2]
        high_curr = df['High'].iloc[i]
        high_prev = df['High'].iloc[i - 1]
        low_curr = df['Low'].iloc[i]

        if low_prev > high_curr:
            fvg_zones.append({
                'index': df.index[i],
                'type': 'FVG_down',
                'gap': (high_curr, low_prev)
            })
        elif high_prev < low_curr:
            fvg_zones.append({
                'index': df.index[i],
                'type': 'FVG_up',
                'gap': (high_prev, low_curr)
            })
    return fvg_zones


# ======= Order Blocks (OB) =======
def detect_order_blocks(df: pd.DataFrame, lookback=10):
    ob_zones = []
    for i in range(lookback, len(df)):
        block_candle = df.iloc[i - lookback]
        body_size = abs(block_candle['Open'] - block_candle['Close'])
        wick_size = abs(block_candle['High'] - block_candle['Low']) - body_size

        if body_size > wick_size:  # сильная свеча, можно считать Order Block
            ob_type = 'OB_bullish' if block_candle['Close'] > block_candle['Open'] else 'OB_bearish'
            ob_zones.append({
                'index': df.index[i - lookback],
                'type': ob_type,
                'zone': (block_candle['Open'], block_candle['Close'])
            })
    return ob_zones


# ======= Liquidity Zones (Equal Highs/Lows) =======
def detect_liquidity_zones(df: pd.DataFrame, threshold=0.0005):
    zones = []
    for i in range(2, len(df)):
        if abs(df['High'].iloc[i] - df['High'].iloc[i - 1]) < threshold:
            zones.append({'index': df.index[i], 'type': 'equal_highs', 'level': df['High'].iloc[i]})
        if abs(df['Low'].iloc[i] - df['Low'].iloc[i - 1]) < threshold:
            zones.append({'index': df.index[i], 'type': 'equal_lows', 'level': df['Low'].iloc[i]})
    return zones
