# advanced_ai_signal_bot.py

import os
import asyncio
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from ta.momentum import RSIIndicator
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from telegram import Bot
from telegram.constants import ParseMode

# === Конфигурация ===
MODEL_FILENAME = 'forex_model.h5'
LOG_FILENAME = 'log.txt'
SCOPES = ['https://www.googleapis.com/auth/drive']
DRIVE_FOLDER_ID = '12GYefwcJwyo4mI4-MwdZzeLZrCAD1I09'
TIMEFRAMES = {'15m': 7, '30m': 14, '1h': 30, '4h': 60}
CONFIDENCE_THRESHOLD = 0.8
SYMBOLS = ['EURUSD=X', 'XAUUSD=X']

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

bot = Bot(token=TELEGRAM_TOKEN)

# === Авторизация Google Drive ===
credentials = service_account.Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

# === Логирование ===
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, format='%(asctime)s - %(message)s')

async def send_telegram_message(text):
    if TELEGRAM_TOKEN and CHAT_ID:
        await bot.send_message(chat_id=CHAT_ID, text=text, parse_mode=ParseMode.HTML)

async def send_telegram_photo(photo_path, caption):
    if TELEGRAM_TOKEN and CHAT_ID:
        with open(photo_path, 'rb') as photo:
            await bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=caption)

def upload_to_drive(filename):
    file_metadata = {'name': filename, 'parents': [DRIVE_FOLDER_ID]}
    media = MediaFileUpload(filename, resumable=True)
    drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

def download_model():
    query = f"'{DRIVE_FOLDER_ID}' in parents and name='{MODEL_FILENAME}'"
    results = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    if not items:
        return False
    request = drive_service.files().get_media(fileId=items[0]['id'])
    with open(MODEL_FILENAME, 'wb') as f:
        f.write(request.execute())
    return True

def preprocess_data(data):
    data = data.dropna()
    close = data['Close'].values.flatten()
    rsi = RSIIndicator(close=close).rsi()
    data['rsi'] = rsi
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
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    return model

def detect_smart_zones(prices):
    prices = pd.Series(prices.flatten())
    highs, lows, zones = deque(maxlen=20), deque(maxlen=20), []
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            highs.append((i, prices[i]))
        elif prices[i] < prices[i-1]:
            lows.append((i, prices[i]))
        if len(highs) >= 2 and highs[-1][1] > highs[-2][1]:
            zones.append(('HH', highs[-1][0]))
        if len(lows) >= 2 and lows[-1][1] < lows[-2][1]:
            zones.append(('LL', lows[-1][0]))
    return zones

def detect_fvg(data):
    gaps = []
    for i in range(2, len(data)):
        prev_high = data['High'].iloc[i-2]
        prev_low = data['Low'].iloc[i-2]
        cur_low = data['Low'].iloc[i]
        if cur_low > prev_high:
            gaps.append((i, 'FVG'))
    return gaps

def plot_chart(symbol, data, events):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data['Close'].values, label='Цена')
    for label, idx in events[-3:]:
        ax.axvline(x=idx, color='red' if label == 'HH' else 'blue', linestyle='--')
        ax.text(idx, data['Close'].values[idx], label, color='black')
    ax.set_title(f'{symbol} + Zones')
    plt.tight_layout()
    path = f'zones_{symbol.replace("=", "")}.png'
    plt.savefig(path)
    plt.close()
    return path

async def analyze_pair(symbol, interval, days):
    logging.info(f'Анализ {symbol} на таймфрейме {interval}')
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    data = yf.download(symbol, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), interval=interval)

    if data.empty or len(data) < 50:
        await send_telegram_message(f"⚠️ Недостаточно данных по {symbol} {interval}")
        return

    X, y = preprocess_data(data)

    if os.path.exists(MODEL_FILENAME):
        model = load_model(MODEL_FILENAME)
    else:
        model = train_model(X, y)
        model.save(MODEL_FILENAME)
        upload_to_drive(MODEL_FILENAME)

    prediction = model.predict(X[-1:])[0][0]
    confidence = float(prediction)
    direction = "🔼 Покупка" if prediction > 0.5 else "🔽 Продажа"

    events = detect_smart_zones(data['Close'].values)
    fvg = detect_fvg(data)

    caption = f"📊 {symbol} {interval}\nСигнал: {direction}\nУверенность: {confidence:.2%}"
    if events or fvg:
        image_path = plot_chart(symbol, data, events + fvg)
        await send_telegram_photo(image_path, caption)
    elif confidence > CONFIDENCE_THRESHOLD:
        await send_telegram_message(caption)

    model.fit(X, y, epochs=3, batch_size=32, verbose=0)
    model.save(MODEL_FILENAME)
    upload_to_drive(MODEL_FILENAME)

def upload_log():
    if os.path.exists(LOG_FILENAME):
        upload_to_drive(LOG_FILENAME)

async def main():
    if not download_model():
        await send_telegram_message("⚠️ Модель не найдена на Google Drive. Создана новая.")

    await send_telegram_message("🤖 Бот успешно запущен и работает!")

    while True:
        for tf, days in TIMEFRAMES.items():
            for symbol in SYMBOLS:
                try:
                    await analyze_pair(symbol, tf, days)
                except Exception as e:
                    msg = f"❌ Ошибка при анализе {symbol} {tf}: {str(e)}"
                    logging.error(msg)
                    await send_telegram_message(msg)
        upload_log()
        await asyncio.sleep(1800)

if __name__ == '__main__':
    asyncio.run(main())
