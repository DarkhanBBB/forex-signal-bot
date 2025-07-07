import numpy as np
import pandas as pd
from smart_money_indicators import (
    detect_bos,
    detect_fvg,
    detect_order_blocks,
    detect_liquidity_zones,
    add_smart_money_indicators,
)

def prepare_data(df):
    try:
        df = df.copy()

        # Применяем Smart Money индикаторы
        df = add_smart_money_indicators(df)

        # Преобразуем индикаторы в признаки для модели
        df.fillna(0, inplace=True)
        features = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'BOS', 'FVG', 'OrderBlock', 'LiquidityZone']
        df = df[features]

        # Нормализация (опционально)
        df = (df - df.min()) / (df.max() - df.min() + 1e-9)

        X = df.values
        y = np.roll(df['Close'].values, -1)  # следующий Close как целевая переменная
        return X[:-1], y[:-1]

    except Exception as e:
        raise ValueError(f"Ошибка при расчёте Smart Money индикаторов: {e}")

def confidence_score(predicted_price, current_price):
    # Уверенность в виде процентного изменения
    return abs((predicted_price - current_price) / current_price)

def should_enter_trade(predicted_price, current_price, threshold=0.008):  # 0.8%
    score = confidence_score(predicted_price, current_price)
    return score > threshold
