import os
import pickle
import pandas as pd
import numpy as np

def load_data_history(file_path='data_history.pkl'):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return []
    return []

def save_data_history(history, file_path='data_history.pkl'):
    with open(file_path, 'wb') as f:
        pickle.dump(history, f)

def append_new_data(history, new_X, new_y):
    if not isinstance(history, list):
        history = []
    if isinstance(new_X, pd.DataFrame):
        new_X = new_X.values
    if isinstance(new_y, pd.Series):
        new_y = new_y.values

    history.append((new_X, new_y))
    return history

def get_combined_data(history):
    X_all = []
    y_all = []
    for X, y in history:
        X_all.append(X)
        y_all.append(y)
    if not X_all or not y_all:
        return np.empty((0, 5)), np.empty((0,))
    return np.vstack(X_all), np.concatenate(y_all)