# Базовый образ с Python
FROM python:3.11-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Установка pipenv и обновление pip
RUN pip install --upgrade pip

# Установка зависимостей
COPY requirements.txt .
RUN pip install -r requirements.txt

# Копируем весь проект в контейнер
COPY . /app
WORKDIR /app

# Указываем переменную окружения для TensorFlow
ENV TF_CPP_MIN_LOG_LEVEL=3

# Команда запуска бота
CMD ["python", "ai_signal_loop.py"]
