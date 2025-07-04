FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Устанавливаем системные зависимости для matplotlib и ta
RUN apt-get update && \
    apt-get install -y build-essential libatlas-base-dev libgl1-mesa-glx && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "ai_signal_loop.py"]
