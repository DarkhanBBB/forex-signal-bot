import os
import io
import logging
import numpy as np
import tensorflow as tf
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from trading_utils import prepare_data

# Настройки
MODEL_NAME = "model.keras"
FOLDER_ID = "12GYefwcJwyo4mI4-MwdZzeLZrCAD1I09"

# Авторизация
creds = service_account.Credentials.from_service_account_file(
    "credentials.json", scopes=["https://www.googleapis.com/auth/drive"]
)
drive_service = build("drive", "v3", credentials=creds)


def download_model_from_drive(local_path=MODEL_NAME):
    """Скачивает модель с Google Drive, если она там есть"""
    logging.info("🔍 Ищем модель на Google Drive...")
    response = drive_service.files().list(
        q=f"name='{MODEL_NAME}' and '{FOLDER_ID}' in parents and trashed=false",
        spaces="drive",
        fields="files(id, name)"
    ).execute()
    files = response.get("files", [])

    if not files:
        logging.warning("⚠️ Модель не найдена на Google Drive.")
        return False

    file_id = files[0]["id"]
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(local_path, "wb")
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    logging.info("✅ Модель успешно загружена с Google Drive.")
    return True


def upload_model_to_drive(local_path=MODEL_NAME):
    """Загружает/обновляет модель на Google Drive"""
    # Ищем файл
    response = drive_service.files().list(
        q=f"name='{MODEL_NAME}' and '{FOLDER_ID}' in parents and trashed=false",
        spaces="drive",
        fields="files(id, name)"
    ).execute()
    files = response.get("files", [])

    media = MediaIoBaseUpload(io.FileIO(local_path, "rb"), mimetype="application/octet-stream")

    if files:
        file_id = files[0]["id"]
        drive_service.files().update(fileId=file_id, media_body=media).execute()
        logging.info("✅ Модель обновлена на Google Drive.")
    else:
        file_metadata = {"name": MODEL_NAME, "parents": [FOLDER_ID]}
        drive_service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        logging.info("✅ Модель загружена на Google Drive.")


def create_model(input_shape):
    """Создаёт простую нейросеть"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_or_load_model(x_train, y_train, model_path=MODEL_NAME):
    """Загружает модель из Google Drive или обучает новую"""
    if not os.path.exists(model_path):
        found = download_model_from_drive(model_path)
        if not found:
            logging.info("🧠 Создаём новую модель.")
            model = create_model(x_train.shape[1:])
        else:
            model = tf.keras.models.load_model(model_path)
    else:
        model = tf.keras.models.load_model(model_path)

    logging.info("🔁 Начинаем дообучение модели.")
    model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0)
    model.save(model_path)
    upload_model_to_drive(model_path)

    logging.info("✅ Модель готова к использованию.")
    return model