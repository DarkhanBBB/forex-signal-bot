import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Путь к файлу с историческими данными
HISTORY_PATH = 'training_data.npz'

# Создание простой нейросети
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Загрузка модели из файла
def load_model(model_path):
    if os.path.exists(model_path):
        return keras_load_model(model_path)
    return None

# Сохранение модели
def save_model(model, model_path):
    model.save(model_path)

# Сохранение обучающих данных в историю
def save_training_history(X, y):
    try:
        if os.path.exists(HISTORY_PATH):
            old = np.load(HISTORY_PATH)
            if 'X' in old and 'y' in old:
                X_old, y_old = old['X'], old['y']
                X_combined = np.vstack([X_old, X])
                y_combined = np.concatenate([y_old, y])
            else:
                X_combined, y_combined = X, y
        else:
            X_combined, y_combined = X, y
        np.savez_compressed(HISTORY_PATH, X=X_combined, y=y_combined)
    except Exception as e:
        print(f"⚠️ Ошибка сохранения истории обучения: {e}")

# Загрузка обучающих данных из истории
def load_training_history():
    try:
        if os.path.exists(HISTORY_PATH):
            data = np.load(HISTORY_PATH)
            if 'X' in data and 'y' in data:
                return data['X'], data['y']
        return None, None
    except Exception as e:
        print(f"⚠️ Ошибка загрузки истории обучения: {e}")
        return None, None

# Обучение модели (или дообучение)
def train_model(model, X_new, y_new, epochs=10):
    save_training_history(X_new, y_new)
    X_total, y_total = load_training_history()

    if X_total is None or y_total is None:
        X_total, y_total = X_new, y_new

    model.fit(X_total, y_total, epochs=epochs, verbose=0)
    return model