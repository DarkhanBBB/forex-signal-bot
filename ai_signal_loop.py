import os
import time
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from ta.momentum import RSIIndicator
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import telegram
from collections import deque
import logging

# === Конфигурация ===
MODEL_FILENAME = 'forex_model.h5'
LOG_FILENAME = 'logs.txt'
SCOPES = ['https://www.googleapis.com/auth/drive']
DRIVE_FOLDER_ID = '12GYefwcJwyo4mI4-MwdZzeLZrCAD1I09'

# === Настройки Telegram ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# === Настройки анализа ===
TIMEFRAMES = {
    '15m': 15,
    '30m': 30,
    '60m': 60,
    '240m': 240
}
CONFIDENCE_THRESHOLD = 0.8
SYMBOLS = ['EURUSD=X', 'XAUUSD=X']
STARTUP_MESSAGE_SENT = False

# === Авторизация Google Drive ===
credentials = service_account.Credentials.from_service_account_file(
    'credentials.json', scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

# === Telegram бот ===
bot = telegram.Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and CHAT_ID else None

# === Настройка логгера ===
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def send_telegram_message(text):
    if bot:
        import asyncio
        asyncio.run(bot.send_message(chat_id=CHAT_ID, text=text))
    else:
        print("❌ Telegram переменные окружения не заданы.")


def send_telegram_image(path, caption):
    if bot:
        import asyncio
        from telegram import InputFile
        with open(path, 'rb') as img:
            asyncio.run(bot.send_photo(chat_id=CHAT_ID, photo=InputFile(img), caption=caption))


def upload_model(service):
    media = MediaFileUpload(MODEL_FILENAME, resumable=True)
    file_metadata = {'name': MODEL_FILENAME, 'parents': [DRIVE_FOLDER_ID]}
    service.files().create(body=file_metadata, media_body=media, fields='id').execute()


def upload_logs(service):
    media = MediaFileUpload(LOG_FILENAME, resumable=True)
    file_metadata = {'name': LOG_FILENAME, 'parents': [DRIVE_FOLDER_ID]}
    service.files().create(body=file_metadata, media_body=media, fields='id').execute()


def download_model(service):
    results = service.files().list(q=f"'{DRIVE_FOLDER_ID}' in parents and name='{MODEL_FILENAME}'",
                                   spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    if not items:
        print("⚠️ Модель не найдена на Google Drive.")
        return False
    request = service.files().get_media(fileId=items[0]['id'])
    with open(MODEL_FILENAME, 'wb') as f:
        f.write(request.execute())
    return True


def preprocess_data(data):
    data = data.dropna()
    close = data['Close']
    data['rsi'] = RSIIndicator(close=close).rsi()
    data.dropna(inplace=True)
    X = data[['Close', 'rsi']].values
    y = (data['Close'].shift(-1) > data['Close']).astype(int).dropna().values
    X = X[:-1]
    return np.array(X), np.array(y)


def train_model(X, y):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
    return model


def detect_market_structure(prices, lookback=20):
    highs = deque(maxlen=lookback)
    lows = deque(maxlen=lookback)
    bos_events = []
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            highs.append(prices[i])
        elif prices[i] < prices[i - 1]:
            lows.append(prices[i])
        if len(highs) >= 2 and highs[-1] > highs[-2]:
            bos_events.append((i, 'HH'))
        if len(lows) >= 2 and lows[-1] < lows[-2]:
            bos_events.append((i, 'LL'))
    return bos_events


def plot_chart(data, symbol, timeframe, bos=None):
    plt.figure(figsize=(10, 4))
    plt.plot(data['Close'], label='Close Price')
    if bos:
        for i, event in bos:
            plt.axvline(x=i, color='green' if event == 'HH' else 'red', linestyle='--', alpha=0.7)
            plt.text(i, data['Close'].iloc[i], event, fontsize=9)
    plt.title(f'{symbol} {timeframe} Close with BOS')
    plt.legend()
    filename = f"chart_{symbol}_{timeframe}.png".replace("=", "")
    plt.savefig(filename)
    plt.close()
    return filename


def analyze_pair(symbol, timeframe, interval):
    try:
        print(f"\n📊 Анализ {symbol} на {timeframe}...")
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval)

        if data.empty or len(data) < 50:
            print("⚠️ Недостаточно данных.")
            return

        X, y = preprocess_data(data)
        bos_events = detect_market_structure(data['Close'].values)

        if os.path.exists(MODEL_FILENAME):
            model = load_model(MODEL_FILENAME)
        else:
            model = train_model(X, y)

        y_pred = model.predict(X[-1:])[0][0]
        confidence = float(y_pred)
        print(f"🔍 Уверенность: {confidence:.2%}")

        if confidence > CONFIDENCE_THRESHOLD:
            direction = "🔼 Покупка" if confidence > 0.5 else "🔽 Продажа"
            message = f"📈 {symbol} [{timeframe}]\nСигнал: {direction}\nУверенность: {confidence:.2%}"
            send_telegram_message(message)
            chart = plot_chart(data, symbol, timeframe, bos_events)
            send_telegram_image(chart, f"📊 {symbol} {timeframe} сигнал")

        model.fit(X, y, epochs=3, batch_size=32, verbose=0)
        model.save(MODEL_FILENAME)
        if drive_service:
            upload_model(drive_service)
            upload_logs(drive_service)
        logging.info(f"✅ Анализ {symbol} {timeframe} завершён. Уверенность: {confidence:.2%}")
    except Exception as e:
        logging.error(f"❌ Ошибка при анализе {symbol} {timeframe}: {str(e)}")
        send_telegram_message(f"❌ Ошибка при анализе {symbol} {timeframe}: {str(e)}")


# === Главный цикл ===
if __name__ == '__main__':
    if drive_service:
        download_model(drive_service)

    if not STARTUP_MESSAGE_SENT:
        send_telegram_message("🤖 Бот успешно запущен и работает!")
        STARTUP_MESSAGE_SENT = True

    while True:
        for tf, minutes in TIMEFRAMES.items():
            interval = f"{minutes}m"
            for sym in SYMBOLS:
                analyze_pair(sym, tf, interval)
        time.sleep(1800)  # 30 минут
