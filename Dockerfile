# Базовый образ Python 3.10
FROM python:3.10-slim

# Установка зависимостей системы
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && apt-get clean

# Установка рабочей директории
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Установка Python-зависимостей
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Запуск бота
CMD ["python", "ai_signal_loop.py"]
