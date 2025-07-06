import asyncio
import datetime
import logging
import os
import traceback

import numpy as np
import pandas as pd
import tensorflow as tf
yf = __import__('yfinance')

from telegram import Bot

from model_utils import load_model, save_model, train_model, create_model
from trading_utils import detect_bos, detect_fvg, detect_order_blocks, calculate_tp_sl

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Настройки
SYMBOLS = ['EURUSD=X', 'XAUUSD=X']
TIMEFRAMES = ['15m', '30m', '1h', '4h']
MODEL_PATH = 'model.h5'
TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

bot = Bot(token=TOKEN)

def send_telegram_message(message):
    try:
        bot.send_message(chat_id=CHAT_ID, text=message)
    except Exception as e:
        logger.error(f"Ошибка отправки в Telegram: {e}")

def prepare_data(data):
    df = data.copy()
    df = df.dropna()
    df['Return'] = df['Close'].pct_change()
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
    df = df.dropna()
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    y = df['Target'].values
    return X, y

def analyze_symbol(symbol, interval):
    try:
        logger.info(f"Анализ {symbol} на таймфрейме {interval}")
        send_telegram_message(f"Анализ {symbol} на таймфрейме {interval}")

        end = datetime.datetime.utcnow()
        start = end - datetime.timedelta(days=30)

        data = yf.download(symbol, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), interval=interval)
        if data is None or data.empty:
            raise ValueError("Нет данных")

        data = data[~data.index.duplicated(keep='first')]

        # Smart Money элементы анализа
        bos_events = detect_bos(data['Close'])
        fvg_zones = detect_fvg(data)
        order_blocks = detect_order_blocks(data)

        X, y = prepare_data(data)
        if X.shape[0] < 50:
            raise ValueError("Недостаточно данных для анализа")

        model = load_model(MODEL_PATH)
        if model is None:
            model = create_model(X.shape[1])

        model = train_model(model, X, y)
        save_model(model, MODEL_PATH)

        prediction = model.predict(X[-1].reshape(1, -1))[0][0]
        confidence = round(float(prediction) * 100, 2)

        if confidence > 80:
            direction = "BUY" if prediction > 0.5 else "SELL"
            entry = data['Close'].iloc[-1]
            tp, sl = calculate_tp_sl(entry, direction)
            message = (
                f"\u2705 Сигнал для {symbol} ({interval}): {direction}\n"
                f"Уверенность: {confidence}%\n"
                f"TP: {tp:.5f} | SL: {sl:.5f}\n"
                f"Break of Structure: {len(bos_events)}\n"
                f"Order Blocks: {len(order_blocks)} | FVG: {len(fvg_zones)}"
            )
            logger.info(message)
            send_telegram_message(message)

    except Exception as e:
        error_text = f"\u274C Ошибка при анализе {symbol} {interval}: {e}"
        logger.error(error_text)
        send_telegram_message(error_text)
        traceback.print_exc()

async def main():
    first_run = True
    while True:
        if first_run:
            send_telegram_message("\ud83e\udd16 Бот запущен и работает")
            first_run = False

        for interval in TIMEFRAMES:
            for symbol in SYMBOLS:
                await asyncio.to_thread(analyze_symbol, symbol, interval)

        try:
            await asyncio.sleep(1800)  # 30 минут
        except asyncio.CancelledError:
            logger.warning("Цикл прерван")
            break

if __name__ == '__main__':
    asyncio.run(main())
