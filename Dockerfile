# Используем Python-образ с минимальной системой
FROM python:3.10-slim

# Обновляем пакеты и устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем pip-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в директорию /app
COPY . /app
WORKDIR /app

# Копируем credentials.json внутрь контейнера
COPY credentials.json /app/credentials.json

# Устанавливаем переменную среды для авторизации Google API
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/credentials.json"

# Команда запуска Python-бота
CMD ["python", "ai_signal_loop.py"]
