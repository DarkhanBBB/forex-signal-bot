import os
import asyncio
import logging
from datetime import datetime, timedelta
from telegram import Bot

from trading_utils import prepare_data, confidence_score, should_enter_trade
from model_utils import create_model, load_model, save_model, train_model
from data_utils import load_data_history, save_data_history, append_new_data, get_combined_data
from twelve_data_api import download  # ✅ Новый источник данных

# === Конфигурация ===
MODEL_PATH = "model.keras"
LOG_FILE = "bot.log"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Настройка логгирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()
file_handler = logging.FileHandler(LOG_FILE)
logger.addHandler(file_handler)


# === Анализ конкретной пары на таймфрейме ===
async def analyze_symbol(symbol, interval, model, model_path, bot, chat_id):
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=30)
        new_data = download(symbol, interval, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))

        if new_data.empty:
            logger.warning(f"⚠️ Нет данных по {symbol} ({interval}) — пропуск анализа.")
            return

        X_new, y_new = prepare_data(new_data)

        if len(X_new) == 0:
            logger.warning(f"⚠️ Недостаточно данных после подготовки для {symbol} ({interval})")
            return

        conf = confidence_score(model, X_new)

        message = f"📊 [{symbol}] ({interval})\n📈 Уверенность сигнала: {conf:.2f}"

        if should_enter_trade(conf):
            message += "\n✅ Сигнал на вход в рынок!"
        else:
            message += "\n⛔ Нет сигнала на вход."

        await bot.send_message(chat_id=chat_id, text=message)

        # Обучение модели на новых данных
        X_total, y_total = get_combined_data(symbol, interval)
        model = train_model(model, X_total, y_total, epochs=5)
        save_model(model, model_path)

        # Обновляем историю
        history_df = load_data_history(symbol, interval)
        updated_df = append_new_data(history_df, new_data, X_new, y_new)
        save_data_history(symbol, interval, updated_df)

        logger.info(f"✅ Анализ завершён для {symbol} ({interval})")

    except Exception as e:
        error_text = f"❌ Ошибка при анализе {symbol} {interval}: {e}"
        logger.exception(error_text)
        await bot.send_message(chat_id=chat_id, text=error_text)


# === Основной цикл ===
async def main():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)

    model = load_model(MODEL_PATH)
    if model is None:
        model = create_model(input_dim=8)
        logger.info("Создана новая модель")

    sent_startup_msg = False
    while True:
        try:
            if not sent_startup_msg:
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="🚀 Бот запущен и работает 24/7")
                sent_startup_msg = True

            symbols = ["EUR/USD", "XAU/USD"]
            intervals = ["1h", "4h"]

            for symbol in symbols:
                for interval in intervals:
                    logger.info(f"Анализ {symbol} на таймфрейме {interval}")
                    await analyze_symbol(symbol, interval, model, MODEL_PATH, bot, TELEGRAM_CHAT_ID)

        except Exception as e:
            logger.exception(f"❌ Глобальная ошибка: {e}")
            try:
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"❗ Ошибка в главном цикле: {e}")
            except:
                pass

        await asyncio.sleep(1800)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Цикл прерван")
