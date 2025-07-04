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
from collections import deque
import matplotlib.pyplot as plt

# === Конфигурация ===
MODEL_FILENAME = 'forex_model.h5'
LOG_FILENAME = 'forex_log.txt'
SCOPES = ['https://www.googleapis.com/auth/drive']
DRIVE_FOLDER_ID = '12GYefwcJwyo4mI4-MwdZzeLZrCAD1I09'

# === Настройки Telegram ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# === Настройки анализа ===
TIMEFRAMES = {'15m': 7, '30m': 14, '60m': 30, '240m': 60}  # дни для каждого таймфрейма
CONFIDENCE_THRESHOLD = 0.8
SYMBOLS = ['EURUSD=X', 'XAUUSD=X']
STARTUP_MESSAGE_SENT = False

# === Авторизация Google Drive ===
credentials = service_account.Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

# === Telegram бот ===
bot = telegram.Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and CHAT_ID else None

def send_telegram_message(text):
    if bot:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(bot.send_message(chat_id=CHAT_ID, text=text))
        loop.close()
    else:
        print("❌ Telegram переменные окружения не заданы.")

def save_log_to_drive():
    if os.path.exists(LOG_FILENAME):
        media = MediaFileUpload(LOG_FILENAME, resumable=True)
        file_metadata = {'name': LOG_FILENAME, 'parents': [DRIVE_FOLDER_ID]}
        drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

def log_event(message):
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILENAME, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")


def upload_model(service):
    media = MediaFileUpload(MODEL_FILENAME, resumable=True)
    file_metadata = {'name': MODEL_FILENAME, 'parents': [DRIVE_FOLDER_ID]}
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
    data['rsi'] = RSIIndicator(close=close.values.reshape(-1)).rsi()
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
        if prices[i] > prices[i-1]:
            highs.append(prices[i])
        elif prices[i] < prices[i-1]:
            lows.append(prices[i])
        if len(highs) >= 2 and highs[-1] > highs[-2]:
            bos_events.append((i, 'HH'))
        if len(lows) >= 2 and lows[-1] < lows[-2]:
            bos_events.append((i, 'LL'))
    return bos_events

def save_bos_plot(data, bos_events, symbol, interval):
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label='Close')
    for idx, bos_type in bos_events:
        plt.scatter(idx, data['Close'].iloc[idx], label=bos_type, color='red' if bos_type == 'LL' else 'green')
    plt.title(f'{symbol} - BOS ({interval})')
    plt.legend()
    filename = f'bos_{symbol.replace("=", "")}_{interval}.png'
    plt.savefig(filename)
    plt.close()
    return filename

def analyze_pair(symbol, interval):
    print(f"\n📊 Анализ {symbol} [{interval}]...")
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=TIMEFRAMES[interval])
    data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=interval)

    if data.empty or len(data) < 50:
        print("⚠️ Недостаточно данных.")
        return

    X, y = preprocess_data(data)
    bos_events = detect_market_structure(data['Close'].values)
    if bos_events:
        bos_time = data.index[bos_events[-1][0]]
        bos_type = bos_events[-1][1]
        print(f"📉 BOS обнаружен: {bos_type} на {bos_time}")
        image_file = save_bos_plot(data, bos_events, symbol, interval)
        bot.send_photo(chat_id=CHAT_ID, photo=open(image_file, 'rb'))

    if os.path.exists(MODEL_FILENAME):
        model = load_model(MODEL_FILENAME)
    else:
        model = train_model(X, y)

    y_pred = model.predict(X[-1:])[0][0]
    confidence = float(y_pred)
    print(f"🔍 Уверенность: {confidence:.2%}")

    if confidence > CONFIDENCE_THRESHOLD:
        direction = "🔼 Покупка" if confidence > 0.5 else "🔽 Продажа"
        message = f"📈 {symbol} [{interval}]\nСигнал: {direction}\nУверенность: {confidence:.2%}"
        send_telegram_message(message)

    model.fit(X, y, epochs=3, batch_size=32, verbose=0)
    model.save(MODEL_FILENAME)
    upload_model(drive_service)
    log_event(f"✅ Дообучение модели для {symbol} завершено.")
    send_telegram_message(f"✅ Модель для {symbol} [{interval}] дообучена и обновлена.")

# === Главный цикл ===
if __name__ == '__main__':
    try:
        if drive_service:
            download_model(drive_service)

        if not STARTUP_MESSAGE_SENT:
            send_telegram_message("🤖 Бот успешно запущен и работает!")
            STARTUP_MESSAGE_SENT = True

        while True:
            for tf in TIMEFRAMES:
                for sym in SYMBOLS:
                    try:
                        analyze_pair(sym, tf)
                    except Exception as e:
                        err = f"❌ Ошибка при анализе {sym} [{tf}]: {str(e)}"
                        print(err)
                        send_telegram_message(err)
                        log_event(err)
            save_log_to_drive()
            time.sleep(1800)  # каждые 30 минут
    except Exception as fatal:
        send_telegram_message(f"💥 Критическая ошибка: {str(fatal)}")
        log_event(f"💥 Критическая ошибка: {str(fatal)}")
        save_log_to_drive()
