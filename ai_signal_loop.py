import asyncio
import logging
import os
from datetime import datetime, timedelta
from telegram import Bot
from data_utils import load_data_history, save_data_history, append_new_data, get_combined_data
from model_manager import train_or_load_model
from trading_utils import prepare_data, confidence_score, should_enter_trade
from twelve_data_api import download

# Настройки
SYMBOLS = ["EUR/USD", "XAU/USD"]
TIMEFRAMES = ["1h", "2h", "3h", "4h"]
MODEL_PATH = "model.keras"
LOG_FILE = "bot.log"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Логирование
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()
file_handler = logging.FileHandler(LOG_FILE)
logger.addHandler(file_handler)

async def send_message(bot, text):
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
    except Exception as e:
        logger.error(f"Ошибка при отправке сообщения в Telegram: {e}")

async def analyze_symbol(symbol, interval, bot):
    try:
        logger.info(f"📊 Анализ {symbol} ({interval})")
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=30)

        new_data = download(symbol, interval, start_time, end_time)
        if new_data is None or new_data.empty:
            await send_message(bot, f"⚠️ Нет данных по {symbol} ({interval})")
            return

        X_new, y_new = prepare_data(new_data)
        if len(X_new) == 0:
            await send_message(bot, f"⚠️ Недостаточно данных для анализа {symbol} ({interval})")
            return

        # Обновляем историю
        history = load_data_history(symbol, interval)
        updated = append_new_data(history, new_data, X_new, y_new)
        save_data_history(symbol, interval, updated)

        # Объединяем все данные
        X_total, y_total = get_combined_data(symbol, interval)
        if X_total.shape[0] == 0:
            await send_message(bot, f"❌ Нет объединённых данных по {symbol} ({interval})")
            return

        model = train_or_load_model(X_total, y_total, model_path=MODEL_PATH)
        prediction = model.predict(X_new[-1].reshape(1, *X_new.shape[1:]))[0]
        conf = confidence_score(prediction)
        direction = should_enter_trade(prediction, conf)

        msg = f"📉 {symbol} ({interval})\nУверенность: {conf:.2f}%"
        msg += f"\n💡 Сигнал: {direction.upper()}" if direction else "\n⛔ Сигнала нет"
        await send_message(bot, msg)

    except Exception as e:
        logger.error(f"❌ Ошибка при анализе {symbol} {interval}: {e}")
        await send_message(bot, f"❌ Ошибка при анализе {symbol} ({interval}): {e}")

async def main():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await send_message(bot, "🤖 Бот запущен и работает 24/7")
    while True:
        for symbol in SYMBOLS:
            for interval in TIMEFRAMES:
                await analyze_symbol(symbol, interval, bot)
        await asyncio.sleep(1800)  # каждые 30 минут

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⛔ Бот остановлен вручную.")
