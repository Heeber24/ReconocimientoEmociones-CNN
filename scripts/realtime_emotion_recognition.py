# -*- coding: utf-8 -*-
"""
Reconocimiento de emociones en tiempo real con la cámara.
Carga el modelo entrenado (Transfer ImageNet o CNN desde cero) y muestra la emoción detectada. Paso 5 del flujo.
Ajustar TRANSFER_MODEL_NAME o use_custom_cnn según el modelo que hayas usado. Ver README.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
from keras.models import load_model  # type: ignore

from config import MODELS_DIR, MODEL_CUSTOM, MODEL_FROM_CUSTOM, EMOTION_LIST, FACE_SIZE, IMAGES_ARE_BGR, USE_GRAY_ROI

# Qué modelo cargar (elige uno):
# - Transfer (ImageNet): use_custom_cnn = False, TRANSFER_MODEL_NAME = "EfficientNetB0", etc.
# - CNN desde cero: use_custom_cnn = True  ← si entrenaste con train_cnn_from_scratch
# - Transfer desde tu modelo: use_from_custom = True
TRANSFER_MODEL_NAME = "EfficientNetB0"
use_custom_cnn = True   # True = cargar emotion_recognition_Personal.keras (CNN desde cero)
use_from_custom = False

if use_from_custom:
    model_path = MODEL_FROM_CUSTOM
elif use_custom_cnn:
    model_path = MODEL_CUSTOM
else:
    model_path = MODELS_DIR / f"emotion_recognition_{TRANSFER_MODEL_NAME}_model.keras"

if not model_path.exists():
    print(f"Error: No se encontró el modelo en {model_path}")
    print("Entrena con train_cnn_from_scratch.py (use_custom_cnn=True) o train_transfer_imagenet.py (use_custom_cnn=False).")
    sys.exit(1)
model = load_model(str(model_path))
print("Modelo cargado:", model_path.name)

# Orden alfabético: debe coincidir con las carpetas en train/ (flow_from_directory)
emotion_labels = EMOTION_LIST

camera_index = int(input("Seleccione la cámara (0 = interna, 1 = externa): "))  # Pide al usuario que seleccione la cámara (0 para interna, 1 para externa)
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Inicializa la captura de video desde la cámara seleccionada

if not cap.isOpened():  # Verifica si la cámara se abrió correctamente
    print(f"Error: No se pudo abrir la cámara {camera_index}")  # Imprime un mensaje de error si la cámara no se abrió
    exit()  # Sale del programa

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Carga el clasificador Haar Cascade para detección de rostros

cv2.namedWindow('Reconocimiento de Emociones')  # Crea una ventana con el título 'Reconocimiento de Emociones'

while cap.isOpened():  # Bucle principal para capturar y procesar fotogramas de la cámara
    ret, frame = cap.read()  # Lee un fotograma de la cámara y almacena el resultado en 'ret' y el fotograma en 'frame'
    if not ret:  # Verifica si se leyó el fotograma correctamente
        print("Error al leer el frame de la cámara.")  # Imprime un mensaje de error si no se leyó el fotograma
        break  # Sale del bucle

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convierte el fotograma a escala de grises
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)  # Detecta rostros en el fotograma usando el clasificador Haar Cascade

    for (x, y, w, h) in faces:  # Itera sobre los rostros detectados
        if USE_GRAY_ROI:
            # Recorte en gris para coincidir mejor con FER2013
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, FACE_SIZE, interpolation=cv2.INTER_CUBIC)
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)  # 3 canales (BGR)
        else:
            # Recorte en color (webcam)
            roi = frame[y:y + h, x:x + w]
            roi = cv2.resize(roi, FACE_SIZE, interpolation=cv2.INTER_CUBIC)  # 3 canales BGR

        roi = roi.astype("float32") / 255.0  # Normaliza al rango [0, 1]

        # Si el entrenamiento convertía BGR->RGB, entonces lo hacemos aquí.
        # (Para ROI gris, esta operación no cambia nada porque los canales son iguales.)
        if IMAGES_ARE_BGR:
            roi = roi[..., ::-1].copy()  # BGR -> RGB

        roi = np.expand_dims(roi, axis=0)  # (1, 224, 224, 3)

        try:
            prediction = model.predict(roi, verbose=0)
            emotion_index = int(np.argmax(prediction))
            emotion = emotion_labels[emotion_index]
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error durante la predicción: {e}")

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dibuja un rectángulo alrededor del rostro detectado

    cv2.imshow('Reconocimiento de Emociones', frame)  # Muestra el fotograma con el reconocimiento de emociones

    key = cv2.waitKey(1) & 0xFF  # Espera a que se presione una tecla
    if key == 27 or cv2.getWindowProperty('Reconocimiento de Emociones', cv2.WND_PROP_VISIBLE) < 1:  # Verifica si se presionó la tecla ESC o se cerró la ventana
        break  # Sale del bucle

cap.release()  # Libera los recursos de la cámara
cv2.destroyAllWindows()  # Cierra todas las ventanas