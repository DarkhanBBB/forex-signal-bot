import asyncio
import datetime
import logging
import os
# Отключаем OneDNN и GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import traceback

import numpy as np
import pandas as pd
import tensorflow as tf
from telegram import Bot

from model_utils import load_model, save_model, train_model, create_model
from trading_utils import detect_bos, detect_fvg, detect_order_blocks
from data_utils import load_data_history, save_data_history, append_new_data, get_combined_data
from twelve_data_api import download  # ✅ Новый импорт

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Настройки
SYMBOLS = ['EUR/USD', 'XAU/USD']  # ✅ Обновлённые тикеры
TIMEFRAMES = ['15m', '30m', '1h', '4h']
MODEL_PATH = 'model.h5'
TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

bot = Bot(token=TOKEN)

async def send_telegram_message(message):
    try:
        await bot.send_message(chat_id=CHAT_ID, text=message)
    except Exception as e:
        logger.error(f"Ошибка отправки в Telegram: {e}")

def prepare_data(data):
    df = data.copy()
    df = df.dropna()
    df['Return'] = df['Close'].pct_change()
    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)

    try:
        df['BOS'] = pd.Series(detect_bos(df['Close']), index=df.index)
        df['FVG'] = detect_fvg(df)
        df['OB'] = detect_order_blocks(df)
    except Exception as e:
        raise ValueError(f"Ошибка при расчёте Smart Money индикаторов: {e}")

    df = df.dropna()
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'BOS', 'FVG', 'OB']
    X = df[features].values
    y = df['Target'].values
    return X, y

async def analyze_symbol(symbol, interval):
    try:
        logger.info(f"Анализ {symbol} на таймфрейме {interval}")
        await send_telegram_message(f"Анализ {symbol} на таймфрейме {interval}")

        # Таймфрейм влияет на количество дней
        end = datetime.datetime.utcnow()
        if interval in ['15m', '30m', '1h']:
            start = end - datetime.timedelta(days=6)
        else:
            start = end - datetime.timedelta(days=30)

        # ✅ Загрузка данных через Twelve Data
        new_data = download(symbol, interval, start, end)

        if new_data is None or new_data.empty:
            logger.warning(f"⚠️ Нет данных по {symbol} ({interval}) — пропуск анализа.")
            await send_telegram_message(f"⚠️ Нет данных по {symbol} ({interval}) — пропущено.")
            return

        new_data = new_data[~new_data.index.duplicated(keep='first')]

        history_df = load_data_history(symbol, interval)
        X_new, y_new = prepare_data(new_data)
        updated_df = append_new_data(history_df, new_data, X_new, y_new)
        save_data_history(symbol, interval, updated_df)

        X, y = get_combined_data(symbol, interval)

        if X.shape[0] < 50:
            logger.warning(f"Недостаточно данных для анализа {symbol} ({interval})")
            return

        model = load_model(MODEL_PATH)
        if model is None:
            model = create_model(X.shape[1])

        model = train_model(model, X, y)
        save_model(model, MODEL_PATH)

        prediction = model.predict(X[-1].reshape(1, -1))[0][0]
        confidence = round(float(prediction) * 100, 2)

        if confidence > 80:
            direction = "BUY" if prediction > 0.5 else "SELL"
            message = f"✅ Сигнал для {symbol} ({interval}): {direction}\nУверенность: {confidence}%"
            logger.info(message)
            await send_telegram_message(message)

    except Exception as e:
        error_text = f"❌ Ошибка при анализе {symbol} {interval}: {e}"
        logger.error(error_text)
        await send_telegram_message(error_text)
        traceback.print_exc()

async def main():
    first_run = True
    while True:
        if first_run:
            await send_telegram_message("🤖 Бот запущен и работает")
            first_run = False

        for interval in TIMEFRAMES:
            for symbol in SYMBOLS:
                await analyze_symbol(symbol, interval)

        try:
            await asyncio.sleep(1800)
        except asyncio.CancelledError:
            logger.warning("Цикл прерван")
            break

if __name__ == '__main__':
    asyncio.run(main())
