FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y gcc g++ libglib2.0-0 libsm6 libxrender1 libxext6 && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "ai_signal_loop.py"]
