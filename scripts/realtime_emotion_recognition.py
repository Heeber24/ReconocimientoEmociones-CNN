# -*- coding: utf-8 -*-
"""
Reconocimiento de emociones en tiempo real con la cámara.
Carga el modelo entrenado (Transfer o custom_cnn) y muestra la emoción detectada. Paso 5 del flujo.
Ajustar TRANSFER_MODEL_NAME o use_custom_cnn según el modelo que hayas usado. Ver README.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
from keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore

from config import MODELS_DIR, MODEL_CUSTOM, MODEL_FROM_CUSTOM, EMOTION_LIST

# Qué modelo cargar (elige uno):
# - Transfer (ImageNet): TRANSFER_MODEL_NAME = "EfficientNetB0", "VGG16", etc.
# - CNN desde cero: use_custom_cnn = True
# - Transfer desde tu modelo (transfer_from_custom.py): use_from_custom = True
TRANSFER_MODEL_NAME = "EfficientNetB0"
use_custom_cnn = False
use_from_custom = False  # True = modelo guardado por transfer_from_custom.py

if use_from_custom:
    model_path = MODEL_FROM_CUSTOM
elif use_custom_cnn:
    model_path = MODEL_CUSTOM
else:
    model_path = MODELS_DIR / f"emotion_recognition_{TRANSFER_MODEL_NAME}_model.keras"
model = load_model(str(model_path))

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
        roi_gray = gray[y:y + h, x:x + w]  # Extrae la región de interés (ROI) del rostro en escala de grises
        roi_gray = cv2.resize(roi_gray, (224, 224), interpolation=cv2.INTER_CUBIC)  # Redimensiona el ROI a 224x224 píxeles
        roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)  # Convierte el ROI a color BGR
        roi = roi_color.astype("float") / 255.0  # Normaliza los valores de los píxeles del ROI al rango [0, 1]
        roi = img_to_array(roi)  # Convierte el ROI a un arreglo NumPy
        roi = np.expand_dims(roi, axis=0)  # Agrega una dimensión al arreglo para que coincida con la entrada del modelo

        if roi.shape == (1, 224, 224, 3):  # Verifica si la forma del ROI es correcta (1, 224, 224, 3)
            try:
                prediction = model.predict(roi, verbose=0)  # Realiza la predicción de la emoción usando el modelo cargado (verbose=0 para evitar mensajes innecesarios)
                emotion_index = np.argmax(prediction)  # Obtiene el índice de la emoción con mayor probabilidad
                emotion = emotion_labels[emotion_index]  # Obtiene la etiqueta de la emoción correspondiente al índice
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Muestra la emoción en el fotograma
            except Exception as e:  # Captura cualquier excepción que ocurra durante la predicción
                print(f"Error durante la predicción: {e}")  # Imprime un mensaje de error si ocurre una excepción
        else:  # Si la forma del ROI no es correcta
            print(f"Error: Forma de roi incorrecta: {roi.shape}")  # Imprime un mensaje de error indicando la forma incorrecta del ROI

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dibuja un rectángulo alrededor del rostro detectado

    cv2.imshow('Reconocimiento de Emociones', frame)  # Muestra el fotograma con el reconocimiento de emociones

    key = cv2.waitKey(1) & 0xFF  # Espera a que se presione una tecla
    if key == 27 or cv2.getWindowProperty('Reconocimiento de Emociones', cv2.WND_PROP_VISIBLE) < 1:  # Verifica si se presionó la tecla ESC o se cerró la ventana
        break  # Sale del bucle

cap.release()  # Libera los recursos de la cámara
cv2.destroyAllWindows()  # Cierra todas las ventanas