import os
import time
import asyncio
import logging
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from telegram import Bot
from telegram.constants import ParseMode
from trading_utils import prepare_data, confidence_score, should_enter_trade
from twelve_data_api import download
from model_manager import train_or_load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MODEL_PATH = "model.keras"

bot = Bot(token=TELEGRAM_TOKEN)


def detect_bos(df: pd.DataFrame):
    bos = []
    for i in range(2, len(df)):
        if df["high"].iloc[i - 1] > df["high"].iloc[i - 2] and df["high"].iloc[i] < df["high"].iloc[i - 1]:
            bos.append({"type": "BOS_down", "index": i})
        elif df["low"].iloc[i - 1] < df["low"].iloc[i - 2] and df["low"].iloc[i] > df["low"].iloc[i - 1]:
            bos.append({"type": "BOS_up", "index": i})
    return bos


def detect_fvg(df: pd.DataFrame):
    fvg = []
    for i in range(2, len(df)):
        if df["low"].iloc[i] > df["high"].iloc[i - 2]:
            fvg.append({"type": "FVG_up", "index": i})
        elif df["high"].iloc[i] < df["low"].iloc[i - 2]:
            fvg.append({"type": "FVG_down", "index": i})
    return fvg


def detect_order_blocks(df: pd.DataFrame):
    ob = []
    for i in range(2, len(df)):
        if df["close"].iloc[i - 2] > df["open"].iloc[i - 2] and df["close"].iloc[i - 1] < df["open"].iloc[i - 1]:
            ob.append({"type": "OB_bearish", "index": i - 1})
        elif df["close"].iloc[i - 2] < df["open"].iloc[i - 2] and df["close"].iloc[i - 1] > df["open"].iloc[i - 1]:
            ob.append({"type": "OB_bullish", "index": i - 1})
    return ob


def detect_liquidity_zones(df: pd.DataFrame):
    liq = []
    for i in range(2, len(df)):
        if df["high"].iloc[i] == df["high"].iloc[i - 1]:
            liq.append({"type": "equal_highs", "index": i})
        elif df["low"].iloc[i] == df["low"].iloc[i - 1]:
            liq.append({"type": "equal_lows", "index": i})
    return liq


async def analyze_symbol(symbol, interval, model_path, bot, chat_id):
    logging.info(f"Анализ {symbol} на таймфрейме {interval}")

    try:
        end = pd.Timestamp.utcnow()
        start = end - pd.Timedelta(days=30)
        df = download(symbol, interval, start=start, end=end)

        if df is None or df.empty:
            logging.warning(f"⚠️ Нет данных по {symbol} ({interval}) — пропуск анализа.")
            return

        x, y, _ = prepare_data(df)
        model = train_or_load_model(x, y, model_path)
        prediction = model.predict(x[-1:])[0][0]

        logging.info(f"Уверенность по {symbol} ({interval}): {prediction:.2f}")

        if prediction >= 0.8 and should_enter_trade(prediction):
            message = f"📈 Сигнал для {symbol} ({interval})\nУверенность: {prediction:.2%}"
            await bot.send_message(chat_id=chat_id, text=message)

        bos = detect_bos(df)
        fvg = detect_fvg(df)
        ob = detect_order_blocks(df)
        liq = detect_liquidity_zones(df)
        logging.info(f"📊 BOS: {len(bos)}, FVG: {len(fvg)}, OB: {len(ob)}, LIQ: {len(liq)}")

    except Exception as e:
        logging.error(f"❌ Ошибка при анализе {symbol} {interval}: {e}")
        await bot.send_message(chat_id=chat_id, text=f"❌ Ошибка при анализе {symbol} {interval}: {e}")


async def main():
    symbols = ["EUR/USD", "XAU/USD"]
    intervals = ["1h", "4h"]

    await bot.send_message(chat_id=CHAT_ID, text="🤖 Бот успешно запущен")

    while True:
        for symbol in symbols:
            for interval in intervals:
                await analyze_symbol(symbol, interval, MODEL_PATH, bot, CHAT_ID)
        await asyncio.sleep(1800)


if __name__ == "__main__":
    asyncio.run(main())