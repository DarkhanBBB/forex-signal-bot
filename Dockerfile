# Используем официальный Python-образ
FROM python:3.10-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc g++ \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libatlas-base-dev libjpeg-dev zlib1g-dev \
    libffi-dev libssl-dev git curl unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY . /app

# Обновляем pip и устанавливаем зависимости
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Команда запуска бота
CMD ["python", "ai_signal_loop.py"]
