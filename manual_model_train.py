# manual_model_train.py
import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from ta.momentum import RSIIndicator
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# === Константы ===
MODEL_FILENAME = 'forex_model.h5'
DRIVE_FOLDER_ID = '12GYefwcJwyo4mI4-MwdZzeLZrCAD1I09'
SCOPES = ['https://www.googleapis.com/auth/drive']

# === Авторизация в Google Drive ===
credentials = service_account.Credentials.from_service_account_file(
    'credentials.json', scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

# === Загрузка исторических данных ===
symbol = 'EURUSD=X'
end = datetime.utcnow()
start = end - timedelta(days=30)
data = yf.download(symbol, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), interval='15m')

if data.empty or len(data) < 100:
    print("Недостаточно данных для обучения.")
    exit()

# === Подготовка данных ===
from trading_utils import detect_bos, detect_fvg, detect_order_blocks
data['BOS'] = detect_bos(data['Close'])
data['FVG'] = detect_fvg(data)
data['OB'] = detect_order_blocks(data)
data.dropna(inplace=True)
X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'BOS', 'FVG', 'OB']].dropna().values
y = data['Close'].pct_change().shift(-1)
y = (y > 0).astype(int)
y = y[-len(X):].values
y = y[:len(X)]

# === Обучение модели ===
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# === Сохранение модели ===
model.save(MODEL_FILENAME)

# === Загрузка на Google Drive ===
media = MediaFileUpload(MODEL_FILENAME, resumable=True)
file_metadata = {'name': MODEL_FILENAME, 'parents': [DRIVE_FOLDER_ID]}
drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

print("✅ Модель создана и загружена в Google Drive.")