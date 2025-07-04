import os
import json
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ta
from datetime import datetime, timedelta
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io

# === 1. Telegram-настройки ===
BOT_TOKEN = os.getenv("8056190931:AAGG60aFwZN8yb7RPJWTWRzIsMsQDf_N_cE")
CHAT_ID = os.getenv("6736814967")

# === 2. Google Drive авторизация ===
def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds)

def download_model_from_drive(file_id, filename):
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(filename, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

def upload_model_to_drive(file_id, filename):
    service = get_drive_service()
    media = MediaFileUpload(filename, resumable=True)
    service.files().update(fileId=file_id, media_body=media).execute()

# === 3. Настройки модели и Drive ===
MODEL_FILE = "forex_model.h5"
DRIVE_FILE_ID = "12GYefwcJwyo4mI4-MwdZzeLZrCAD1I09"  # заменишь на ID своего файла

# === 4. Загрузка данных ===
def load_data(symbol):
    end = datetime.utcnow()
    start = end - timedelta(days=90)
    df = yf.download(symbol, start=start, end=end, interval='15m')
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
    df['ema'] = ta.trend.EMAIndicator(df['Close'], 20).ema_indicator()
    df['macd'] = ta.trend.MACD(df['Close']).macd_diff()
    df.dropna(inplace=True)
    return df

# === 5. Подготовка данных ===
def preprocess(df):
    features = ['Close', 'rsi', 'ema', 'macd']
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[features])
    X, y = [], []
    for i in range(24, len(data)):
        X.append(data[i-24:i])
        y.append(1 if data[i, 0] > data[i-1, 0] else 0)
    return np.array(X), np.array(y), scaler

# === 6. Обработка инструмента ===
def handle_symbol(symbol):
    df = load_data(symbol)
    X, y, scaler = preprocess(df)

    # Загрузка или обучение модели
    if os.path.exists(MODEL_FILE):
        model = load_model(MODEL_FILE)
    else:
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            Dropout(0.2),
            LSTM(32),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X, y, epochs=3, batch_size=32, verbose=0)
    model.save(MODEL_FILE)

    # Отправка в Google Drive
    upload_model_to_drive(DRIVE_FILE_ID, MODEL_FILE)

    last_seq = X[-1].reshape(1, 24, 4)
    prediction = float(model.predict(last_seq)[0][0])
    direction = 'BUY' if prediction > 0.5 else 'SELL'

    high = df['High'].iloc[-14:].max()
    low = df['Low'].iloc[-14:].min()
    atr = high - low
    close = df['Close'].iloc[-1]

    tp = close + atr * 0.5 if direction == 'BUY' else close - atr * 0.5
    sl = close - atr * 0.5 if direction == 'BUY' else close + atr * 0.5

    return {
        'symbol': symbol,
        'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'direction': direction,
        'entry_price': round(close, 5),
        'take_profit': round(tp, 5),
        'stop_loss': round(sl, 5),
        'confidence': round(prediction, 4)
    }

# === 7. Telegram-уведомление ===
def send_to_telegram(signal):
    msg = f"""📈 <b>Сигнал от нейросети</b>
<b>Инструмент:</b> {signal['symbol']}
<b>Время:</b> {signal['timestamp']}
<b>Сигнал:</b> {signal['direction']}
<b>Цена входа:</b> {signal['entry_price']}
<b>TP:</b> {signal['take_profit']}
<b>SL:</b> {signal['stop_loss']}
<b>Уверенность:</b> {signal['confidence']*100:.2f}%"""

    r = requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
    )
    if r.status_code == 200:
        print("✅ Сообщение отправлено")
    else:
        print("❌ Ошибка Telegram:", r.text)

# === 8. Запуск цикла ===
if __name__ == "__main__":
    for symbol in ['EURUSD=X', 'XAUUSD=X']:
        signal = handle_symbol(symbol)
        if signal['confidence'] >= 0.8:
            send_to_telegram(signal)
        else:
            print(f"🔕 Нет сигнала для {symbol}, уверенность: {signal['confidence']}")
