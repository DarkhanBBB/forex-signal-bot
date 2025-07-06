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
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
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

# === Переменные окружения ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# === Telegram бот ===
bot = Bot(token=TELEGRAM_TOKEN)

# === Авторизация Google Drive ===
credentials = service_account.Credentials.from_service_account_file(
    'credentials.json', scopes=SCOPES)
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
    close = data['Close']
    rsi = RSIIndicator(close=close).rsi()
    atr = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close']).average_true_range()
    obv = OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()

    data['rsi'] = rsi
    data['atr'] = atr
    data['obv'] = obv
    data.dropna(inplace=True)

    X = data[['Close', 'rsi', 'atr', 'obv']].values
    y = (data['Close'].shift(-1) > data['Close']).astype(int).dropna().values
    X = X[:-1]

    return np.array(X), np.array(y)

def train_model(X, y):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    return model

def detect_bos(close: pd.Series):
    highs, lows, bos = deque(maxlen=20), deque(maxlen=20), []
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            highs.append(close[i])
        elif close[i] < close[i - 1]:
            lows.append(close[i])
        if len(highs) >= 2 and highs[-1] > highs[-2]:
            bos.append((i, 'HH'))
        if len(lows) >= 2 and lows[-1] < lows[-2]:
            bos.append((i, 'LL'))
    return bos

def detect_fvg(data):
    gaps = []
    for i in range(2, len(data)):
        if data['Low'].iloc[i] > data['High'].iloc[i - 2]:
            gaps.append((i, 'Bullish FVG'))
        elif data['High'].iloc[i] < data['Low'].iloc[i - 2]:
            gaps.append((i, 'Bearish FVG'))
    return gaps

def detect_order_blocks(data):
    blocks = []
    for i in range(2, len(data)):
        if data['Close'].iloc[i - 1] < data['Open'].iloc[i - 1] and data['Close'].iloc[i] > data['Open'].iloc[i]:
            blocks.append((i - 1, 'Bullish OB'))
        elif data['Close'].iloc[i - 1] > data['Open'].iloc[i - 1] and data['Close'].iloc[i] < data['Open'].iloc[i]:
            blocks.append((i - 1, 'Bearish OB'))
    return blocks

def plot_chart(symbol, data, bos, fvg, ob, sl=None, tp=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Close'].values, label='Цена')

    for idx, label in bos[-3:]:
        ax.axvline(x=idx, color='red' if label == 'HH' else 'blue', linestyle='--')
        ax.text(idx, data['Close'].values[idx], label, color='black')

    for idx, label in fvg[-3:]:
        ax.axvline(x=idx, color='green', linestyle=':')
        ax.text(idx, data['Close'].values[idx], label, color='green')

    for idx, label in ob[-3:]:
        ax.axvline(x=idx, color='purple', linestyle='-.')
        ax.text(idx, data['Close'].values[idx], label, color='purple')

    if sl:
        ax.axhline(y=sl, color='orange', linestyle='--')
        ax.text(len(data) - 1, sl, 'SL', color='orange', ha='right')
    if tp:
        ax.axhline(y=tp, color='green', linestyle='--')
        ax.text(len(data) - 1, tp, 'TP', color='green', ha='right')

    ax.set_title(f'{symbol} + Smart Money Concepts')
    plt.tight_layout()
    image_path = f'smc_{symbol.replace("=", "")}.png'
    plt.savefig(image_path)
    plt.close()
    return image_path

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

    atr = data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min()
    atr_value = atr.iloc[-1] if not atr.isna().all() else 0
    price = data['Close'].values[-1]
    sl = price - atr_value if prediction > 0.5 else price + atr_value
    tp = price + 2 * atr_value if prediction > 0.5 else price - 2 * atr_value

    bos_events = detect_bos(data['Close'])  # это Series, а не ndarray
    fvg_zones = detect_fvg(data)
    order_blocks = detect_order_blocks(data)

    caption = (
        f"📊 {symbol} {interval}\n"
        f"Сигнал: {direction}\n"
        f"Уверенность: {confidence:.2%}\n"
        f"<b>TP</b>: {tp:.5f}\n"
        f"<b>SL</b>: {sl:.5f}"
    )

    if confidence > CONFIDENCE_THRESHOLD:
        chart_path = plot_chart(symbol, data, bos_events, fvg_zones, order_blocks, sl, tp)
        await send_telegram_photo(chart_path, caption)

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
