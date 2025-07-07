import os
import pandas as pd
import numpy as np

# Путь к директории хранения исторических данных
DATA_DIR = "data_history"
os.makedirs(DATA_DIR, exist_ok=True)

def get_history_filepath(symbol, interval):
    return os.path.join(DATA_DIR, f"{symbol.replace('=','')}_{interval}.csv")

def load_data_history(symbol, interval):
    path = get_history_filepath(symbol, interval)
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        return pd.DataFrame()

def save_data_history(symbol, interval, df):
    path = get_history_filepath(symbol, interval)
    df.to_csv(path)

def append_new_data(history_df, new_df, new_X, new_y):
    if len(new_df) < len(new_X) or len(new_df) < len(new_y):
        raise ValueError("Недостаточно данных в new_df для объединения с признаками")

    # Объединяем данные
    df_to_add = new_df.tail(len(new_X)).copy()

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'BOS', 'FVG', 'OB']
    df_features = pd.DataFrame(new_X, columns=features, index=df_to_add.index)
    df_to_add[features] = df_features
    df_to_add['Target'] = new_y[-len(df_to_add):]

    updated_df = pd.concat([history_df, df_to_add])
    updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
    return updated_df

def get_combined_data(symbol, interval):
    df = load_data_history(symbol, interval)
    df = df.dropna()
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'BOS', 'FVG', 'OB']
    if not all(f in df.columns for f in features + ['Target']):
        raise ValueError("Отсутствуют нужные колонки в исторических данных")
    X = df[features].values
    y = df['Target'].values
    return X, y