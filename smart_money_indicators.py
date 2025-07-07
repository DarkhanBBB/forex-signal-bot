# smart_money_indicators.py

import numpy as np

def detect_bos(close_series):
    # Простой пример: обнаружение Break of Structure (BOS)
    bos = np.zeros_like(close_series)
    for i in range(2, len(close_series)-2):
        prev_high = max(close_series[i-2:i])
        next_high = max(close_series[i+1:i+3])
        if close_series[i] > prev_high and close_series[i] > next_high:
            bos[i] = 1
    return bos
