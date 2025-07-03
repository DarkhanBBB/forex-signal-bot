import yfinance as yf
import pandas as pd
import numpy as np
import ta
import time
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta

# === Telegram настройки ===
BOT_TOKEN = '8056190931:AAGG60aFwZN8yb7RPJWTWRzIsMsQDf_N_cE'
CHAT_ID = '6736814967'

def analyze_market():
    try:
        print(f"\n[{datetime.utcnow()}] 🔄 Начинаем анализ рынка...")

        # === Загрузка данных
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=720)
        data = yf.download('EURUSD=X', start=start_date.strftime('%Y-%m-%d'),
                           end=end_date.strftime('%Y-%m-%d'), interval='1h', progress=False)

        if data.empty:
            raise Exception("❌ Данные не загружены.")

        # === Индикаторы
        close_series = data['Close'].squeeze()
        data['rsi'] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()
        data['ema'] = ta.trend.EMAIndicator(close=close_series, window=20).ema_indicator()
        data['macd'] = ta.trend.MACD(close=close_series).macd_diff()
        data.dropna(inplace=True)

        features = ['Close', 'rsi', 'ema', 'macd']
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[features].values)

        # === Создание последовательностей
        def create_sequences(data, window=24):
            X, y = [], []
            for i in range(window, len(data)):
                X.append(data[i - window:i])
                y.append(1 if data[i, 0] > data[i - 1, 0] else 0)
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_data)
        if len(X) == 0:
            raise Exception("❌ Недостаточно данных.")

        X = X.reshape((X.shape[0], X.shape[1], len(features)))

        # === Обучение модели
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        # === Прогноз
        last_seq = scaled_data[-24:].reshape((1, 24, len(features)))
        prediction = float(model.predict(last_seq, verbose=0)[0][0])

        close_price = float(data['Close'].iloc[-1])
        last_high = data['High'].iloc[-14:]
        last_low = data['Low'].iloc[-14:]
        atr_val = (last_high.max() - last_low.min())

        direction = 'BUY' if prediction > 0.5 else 'SELL'
        tp = close_price + atr_val * 0.5 if direction == 'BUY' else close_price - atr_val * 0.5
        sl = close_price - atr_val * 0.5 if direction == 'BUY' else close_price + atr_val * 0.5

        signal = {
            'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'direction': direction,
            'entry_price': round(close_price, 5),
            'take_profit': round(tp, 5),
            'stop_loss': round(sl, 5),
            'confidence': round(prediction, 4)
        }

        print(f"🔍 Уверенность: {signal['confidence'] * 100:.2f}%")

        if signal['confidence'] >= 0.8:
            # === Telegram сообщение
            message = f"""📈 <b>Сигнал от нейросети</b>
<b>Время:</b> {signal['timestamp']}
<b>Сигнал:</b> {signal['direction']}
<b>Цена входа:</b> {signal['entry_price']}
<b>TP:</b> {signal['take_profit']}
<b>SL:</b> {signal['stop_loss']}
<b>Уверенность:</b> {signal['confidence'] * 100:.2f}%"""

            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            }
            r = requests.post(url, json=payload)
            if r.status_code == 200:
                print("✅ Сигнал отправлен в Telegram.")
            else:
                print("❌ Ошибка Telegram:", r.text)

            pd.DataFrame([signal]).to_csv("ai_signal_output.csv", mode='a', header=False, index=False)
        else:
            print("⚠️ Уверенность < 80%, сигнал не отправлен.")

    except Exception as e:
        print("❌ Ошибка в работе бота:", e)

# === Главный цикл ===
while True:
    analyze_market()
    print("⏳ Ожидание 1 час...\n")
    time.sleep(3600)  # ждать 1 час (3600 секунд)
