import os
import time
import json
import logging
import asyncio
import numpy as np
import pandas as pd
import yfinance as yf
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

# === Конфигурация ===
MODEL_FILENAME = 'forex_model.h5'
SCOPES = ['https://www.googleapis.com/auth/drive']
DRIVE_FOLDER_ID = '12GYefwcJwyo4mI4-MwdZzeLZrCAD1I09'
LOG_FILENAME = 'log.txt'

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

TIMEFRAMES = {
    '15m': '7d',
    '30m': '14d',
    '1h': '30d',
    '4h': '60d'
}
CONFIDENCE_THRESHOLD = 0.8
SYMBOLS = ['EURUSD=X', 'XAUUSD=X']

# === Логирование ===
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)

# === Авторизация Google Drive ===
credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

# === Telegram бот ===
bot = telegram.Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN and CHAT_ID else None

async def send_telegram_message(text):
    if bot:
        await bot.send_message(chat_id=CHAT_ID, text=text)

def upload_model():
    media = MediaFileUpload(MODEL_FILENAME, resumable=True)
    file_metadata = {'name': MODEL_FILENAME, 'parents': [DRIVE_FOLDER_ID]}
    drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

def upload_log():
    media = MediaFileUpload(LOG_FILENAME, resumable=True)
    file_metadata = {'name': LOG_FILENAME, 'parents': [DRIVE_FOLDER_ID]}
    drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

def download_model():
    results = drive_service.files().list(q=f"'{DRIVE_FOLDER_ID}' in parents and name='{MODEL_FILENAME}'", spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    if not items:
        return False
    request = drive_service.files().get_media(fileId=items[0]['id'])
    with open(MODEL_FILENAME, 'wb') as f:
        f.write(request.execute())
    return True

def preprocess_data(data):
    close = data['Close']
    data['rsi'] = RSIIndicator(close=data['Close']).rsi()
    data.dropna(inplace=True)
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

def detect_market_structure(prices):
    highs, lows = deque(maxlen=20), deque(maxlen=20)
    bos = []
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]: highs.append(prices[i])
        if prices[i] < prices[i-1]: lows.append(prices[i])
        if len(highs) > 1 and highs[-1] > highs[-2]: bos.append((i, 'HH'))
        if len(lows) > 1 and lows[-1] < lows[-2]: bos.append((i, 'LL'))
    return bos

def plot_bos(data, bos_events, symbol, tf):
    plt.figure(figsize=(10, 4))
    plt.plot(data['Close'], label='Close')
    for i, kind in bos_events:
        color = 'green' if kind == 'HH' else 'red'
        plt.scatter(i, data['Close'].iloc[i], c=color, label=kind)
    plt.title(f'{symbol} {tf} BOS')
    plt.legend()
    img_path = f'bos_{symbol}_{tf}.png'
    plt.savefig(img_path)
    return img_path

async def analyze(symbol, interval):
    try:
        print(f"\n📊 Анализ {symbol} {interval}...")
        start = datetime.utcnow() - timedelta(days=int(TIMEFRAMES[interval][:-1]))
        data = yf.download(symbol, start=start.strftime('%Y-%m-%d'), interval=interval)
        if data.empty or len(data) < 50:
            logging.warning(f"Недостаточно данных для {symbol} {interval}")
            return
        X, y = preprocess_data(data)
        bos = detect_market_structure(data['Close'].values)
        if bos:
            bos_img = plot_bos(data, bos, symbol, interval)
            with open(bos_img, 'rb') as img:
                await bot.send_photo(chat_id=CHAT_ID, photo=img, caption=f"{symbol} {interval} BOS")
        if os.path.exists(MODEL_FILENAME):
            model = load_model(MODEL_FILENAME)
        else:
            model = train_model(X, y)
        confidence = float(model.predict(X[-1:])[0][0])
        if confidence > CONFIDENCE_THRESHOLD:
            direction = "🔼 Покупка" if confidence > 0.5 else "🔽 Продажа"
            await send_telegram_message(f"📈 {symbol} {interval}\nСигнал: {direction}\nУверенность: {confidence:.2%}")
        model.fit(X, y, epochs=3, batch_size=32, verbose=0)
        model.save(MODEL_FILENAME)
        upload_model()
    except Exception as e:
        err = f"❌ Ошибка при анализе {symbol} {interval}: {e}"
        logging.error(err)
        await send_telegram_message(err)

async def main():
    if not download_model():
        await send_telegram_message("⚠️ Модель не найдена на Google Drive.")
    await send_telegram_message("🤖 Бот успешно запущен и работает!")
    while True:
        for symbol in SYMBOLS:
            for tf in TIMEFRAMES.keys():
                await analyze(symbol, tf)
        upload_log()
        await asyncio.sleep(1800)

if __name__ == '__main__':
    asyncio.run(main())
