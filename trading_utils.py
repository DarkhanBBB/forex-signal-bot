import numpy as np
import pandas as pd
from smart_money_indicators import detect_bos, detect_fvg, detect_order_blocks, detect_liquidity_zones
from smart_money_indicators import (
    detect_bos,
    detect_fvg,
    detect_order_blocks,
    detect_liquidity_zones,
    add_smart_money_indicators,
)

def prepare_data(df):
    df = df.copy()

    df["bos"] = detect_bos(df)
    df["fvg"] = detect_fvg(df)
    df["order_blocks"] = detect_order_blocks(df)
    df["liquidity_zones"] = detect_liquidity_zones(df)

    return df

    except Exception as e:
        raise ValueError(f"Ошибка при расчёте Smart Money индикаторов: {e}")

def confidence_score(predicted_price, current_price):
    # Уверенность в виде процентного изменения
    return abs((predicted_price - current_price) / current_price)

def should_enter_trade(predicted_price, current_price, threshold=0.008):  # 0.8%
    score = confidence_score(predicted_price, current_price)
    return score > threshold
