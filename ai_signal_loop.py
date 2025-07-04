import os
import time
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import io
import requests

# === НАСТРОЙКИ ===
SCOPES = ['https://www.googleapis.com/auth/drive']
GDRIVE_FOLDER_ID = '12GYefwcJwyo4mI4-MwdZzeLZrCAD1I09'
MODEL_FILENAME = 'forex_model.h5'
CREDENTIALS_FILE = 'credentials.json'
CHECK_INTERVAL_MINUTES = 30
CONFIDENCE_THRESHOLD = 0.8
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")


# === Авторизация Google Drive ===
def get_drive_service():
    credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=credentials)
    if os.path.exists(MODEL_FILENAME):
    upload_model(drive_service)
else:
    print("⚠️ Модель ещё не обучена, файл не найден.")

def download_model(service):
    query = f"'{GDRIVE_FOLDER_ID}' in parents and name='{MODEL_FILENAME}' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    if not items:
        print("⚠️ Модель не найдена на Google Drive.")
        return False
    file_id = items[0]['id']
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(MODEL_FILENAME, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    print("✅ Модель загружена с Google Drive.")
    return True

def upload_model(service):
    query = f"'{GDRIVE_FOLDER_ID}' in parents and name='{MODEL_FILENAME}' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    media = MediaFileUpload(MODEL_FILENAME, resumable=True)
    if items:
        file_id = items[0]['id']
        service.files().update(fileId=file_id, media_body=media).execute()
    else:
        file_metadata = {'name': MODEL_FILENAME, 'parents': [GDRIVE_FOLDER_ID]}
        service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print("✅ Модель обновлена на Google Drive.")

# === Обработка пары ===
def analyze_pair(ticker):
    print(f"\n📊 Анализ {ticker}...")
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=59)
    data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'),
                       interval='15m', progress=False)

    if data.empty or len(data) < 100:
        print("⚠️ Недостаточно данных.")
        return

    close = data['Close']
    data['rsi'] = RSIIndicator(close).rsi()
    data['ema'] = EMAIndicator(close, window=20).ema_indicator()
    data['macd'] = MACD(close).macd_diff()
    data.dropna(inplace=True)

    features = ['Close', 'rsi', 'ema', 'macd']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features].values)

    def create_sequences(data, window=24):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i - window:i])
            y.append(1 if data[i, 0] > data[i - 1, 0] else 0)
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data)
    if len(X) == 0:
        print("⚠️ Недостаточно данных для обучения.")
        return
    X = X.reshape((X.shape[0], X.shape[1], len(features)))

    # === Обучение или загрузка модели ===
    if os.path.exists(MODEL_FILENAME):
        model = load_model(MODEL_FILENAME)
        model.fit(X, y, epochs=2, batch_size=32, verbose=0)
    else:
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=10, batch_size=32, verbose=1)

    model.save(MODEL_FILENAME)

    # === Прогноз ===
    last_seq = scaled_data[-24:]
    last_seq = last_seq.reshape((1, 24, len(features)))
    prediction = float(model.predict(last_seq, verbose=0)[0][0])
    confidence = round(prediction, 4)

    # === Сигнал ===
    last_high = data['High'].iloc[-14:]
    last_low = data['Low'].iloc[-14:]
    atr_val = (last_high.max() - last_low.min())
    close_price = float(data['Close'].iloc[-1])
    direction = 'BUY' if prediction > 0.5 else 'SELL'
    tp = close_price + atr_val * 0.5 if direction == 'BUY' else close_price - atr_val * 0.5
    sl = close_price - atr_val * 0.5 if direction == 'BUY' else close_price + atr_val * 0.5

    signal = {
        'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'pair': ticker,
        'direction': direction,
        'entry_price': round(close_price, 5),
        'take_profit': round(tp, 5),
        'stop_loss': round(sl, 5),
        'confidence': confidence
    }

    if confidence >= CONFIDENCE_THRESHOLD:
        send_telegram_signal(signal)
    else:
        print(f"ℹ️ Уверенность {confidence*100:.2f}% ниже порога.")

def send_telegram_signal(signal):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("❌ Telegram переменные окружения не заданы.")
        return

    message = f"""📈 <b>Сигнал от нейросети</b>
<b>Пара:</b> {signal['pair']}
<b>Время:</b> {signal['timestamp']}
<b>Сигнал:</b> {signal['direction']}
<b>Цена входа:</b> {signal['entry_price']}
<b>TP:</b> {signal['take_profit']}
<b>SL:</b> {signal['stop_loss']}
<b>Уверенность:</b> {signal['confidence']*100:.2f}%"""

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("✅ Сигнал отправлен в Telegram.")
    else:
        print("❌ Ошибка при отправке в Telegram:", response.text)

# === Основной цикл ===
if __name__ == '__main__':
    drive_service = get_drive_service()
    download_model(drive_service)
    printed_start = False

    while True:
        if not printed_start:
            send_telegram_signal({
                'pair': 'INFO',
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                'direction': 'RUNNING',
                'entry_price': 0,
                'take_profit': 0,
                'stop_loss': 0,
                'confidence': 1.0
            })
            printed_start = True

        for sym in ['EURUSD=X', 'XAUUSD=X']:
            try:
                analyze_pair(sym)
            except Exception as e:
                print(f"❌ Ошибка при анализе {sym}: {e}")

        if model:
    upload_model(drive_service)  # ← ❌ нет отступа — будет ошибка

        print(f"🕒 Пауза {CHECK_INTERVAL_MINUTES} минут...\n")
        time.sleep(CHECK_INTERVAL_MINUTES * 60)
