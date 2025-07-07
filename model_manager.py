import os
import io
import logging
import numpy as np
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# === Google Drive настройки ===
FOLDER_ID = "12GYefwcJwyo4mI4-MwdZzeLZrCAD1I09"
MODEL_FILENAME = "model.keras"

# Авторизация
creds = Credentials.from_service_account_file("service_account.json", scopes=["https://www.googleapis.com/auth/drive"])
drive_service = build("drive", "v3", credentials=creds)

def find_model_file():
    query = f"'{FOLDER_ID}' in parents and name = '{MODEL_FILENAME}' and trashed = false"
    results = drive_service.files().list(q=query, spaces='drive', fields="files(id, name)").execute()
    files = results.get("files", [])
    return files[0] if files else None

def download_model():
    file = find_model_file()
    if not file:
        return None
    request = drive_service.files().get_media(fileId=file["id"])
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    with open(MODEL_FILENAME, "wb") as f:
        f.write(fh.read())
    return load_model(MODEL_FILENAME)

def upload_model():
    file = find_model_file()
    media = MediaIoBaseUpload(open(MODEL_FILENAME, "rb"), mimetype="application/octet-stream")
    if file:
        drive_service.files().update(fileId=file["id"], media_body=media).execute()
    else:
        drive_service.files().create(
            body={"name": MODEL_FILENAME, "parents": [FOLDER_ID]},
            media_body=media,
            fields="id"
        ).execute()

def create_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_or_load_model(X=None, y=None, model_path="model.keras"):
    if os.path.exists(model_path):
        logging.info("📥 Загружаю модель из локального файла...")
        return load_model(model_path)

    logging.info("☁️ Ищу модель в Google Drive...")
    model = download_model()
    if model is not None:
        logging.info("✅ Модель загружена с Google Drive.")
        return model

    if X is None or y is None:
        raise ValueError("Нет обучающих данных для создания новой модели")

    logging.info("🔄 Обучаю новую модель...")
    model = create_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1, callbacks=[EarlyStopping(patience=3)], verbose=1)
    model.save(model_path)
    upload_model()
    logging.info("✅ Модель обучена и загружена в Google Drive.")
    return model