import os
import time
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from ta.momentum import RSIIndicator
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import telegram

# === Конфигурация ===
MODEL_FILENAME = 'forex_model.h5'
SCOPES = ['https://www.googleapis.com/auth/drive']
DRIVE_FOLDER_ID = '12GYefwcJwyo4mI4-MwdZzeLZrCAD1I09'
TIMEFRAME_MINUTES = 15
INTERVAL = f'{TIMEFRAME_MINUTES}m'
CONFIDENCE_THRESHOLD = 0.8
SYMBOLS = ['EURUSD=X', 'XAUUSD=X']

# === Telegram ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
from telegram import Bot
import asyncio
bot = Bot(token=TELEGRAM_TOKEN)

# === Авторизация Google Drive ===
credentials = service_account.Credentials.from_service_account_file(
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS"), scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

def send_telegram_message(text):
    if bot:
        asyncio.run(bot.send_message(chat_id=CHAT_ID, text=text))
    else:
        print("❌ Telegram переменные окружения не заданы.")

def upload_model(service):
    try:
        media = MediaFileUpload(MODEL_FILENAME, resumable=True)
        file_metadata = {'name': MODEL_FILENAME, 'parents': [DRIVE_FOLDER_ID]}
        service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")

def download_model(service):
    try:
        results = service.files().list(
            q=f"'{DRIVE_FOLDER_ID}' in parents and name='{MODEL_FILENAME}'",
            spaces='drive', fields='files(id, name)').execute()
        items = results.get('files', [])
        if not items:
            print("⚠️ Модель не найдена на Google Drive.")
            return False
        request = service.files().get_media(fileId=items[0]['id'])
        with open(MODEL_FILENAME, 'wb') as f:
            f.write(request.execute())
        print("✅ Модель загружена с Google Drive.")
        return True
    except Exception as e:
        print(f"❌ Ошибка при скачивании модели: {e}")
        return False

def preprocess_data(data):
    data = data.dropna()
    data['rsi'] = RSIIndicator(close=data['Close']).rsi()
    data = data.dropna()
    X = data[['Close', 'rsi']].values
    y = (data['Close'].shift(-1) > data['Close']).astype(int).dropna().values
    X = X[:-1]
    return np.array(X), np.array(y)

def train_model(X, y):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model

def analyze_pair(symbol):
    print(f"\n📊 Анализ {symbol}...")
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)

    try:
        data = yf.download(
        symbol,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        interval=INTERVAL
        )
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        return

    if data.empty or len(data) < 50:
        print("⚠️ Недостаточно данных.")
        return

    X, y = preprocess_data(data)

    if os.path.exists(MODEL_FILENAME):
        model = load_model(MODEL_FILENAME)
    else:
        model = train_model(X, y)

    y_pred = model.predict(X[-1:])[0][0]
    confidence = float(y_pred)
    print(f"🔍 Уверенность: {confidence:.2%}")

    if confidence > CONFIDENCE_THRESHOLD:
        direction = "🔼 Покупка" if confidence > 0.5 else "🔽 Продажа"
        message = f"📈 {symbol}\nСигнал: {direction}\nУверенность: {confidence:.2%}"
        send_telegram_message(message)

    model.fit(X, y, epochs=3, batch_size=32, verbose=0)
    model.save(MODEL_FILENAME)
    upload_model(drive_service)

# === Главный цикл ===
if __name__ == '__main__':
    STARTUP_FLAG_PATH = "startup_flag.txt"
    if not os.path.exists(STARTUP_FLAG_PATH):
        send_telegram_message("🤖 Бот успешно запущен и работает!")
        with open(STARTUP_FLAG_PATH, 'w') as f:
            f.write("started")

    download_model(drive_service)

    while True:
        for sym in SYMBOLS:
            try:
                analyze_pair(sym)
            except Exception as e:
                print(f"❌ Ошибка при анализе {sym}: {e}")
        time.sleep(1800)  # каждые 30 минут
