import cv2  # Importa la biblioteca OpenCV para procesamiento de imágenes y video
import numpy as np  # Importa la biblioteca NumPy para operaciones numéricas
import tensorflow as tf  # Importa la biblioteca TensorFlow para aprendizaje profundo
from keras.models import load_model  # Importa la función load_model de Keras para cargar modelos preentrenados # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # Importa la función img_to_array para convertir imágenes en arreglos NumPy # type: ignore
import time  # Importa la biblioteca time para medir el tiempo de ejecución (no se utiliza directamente en este código, pero estaba en el original)

#model_path = r'C:\Users\Havalos\Desktop\CEA\Maestria_Inteligencia_Artificial\Sistemas Cognitivos Artificiales\Reconocimiento_Emociones\models\emotion_recognition_EfficientNetB0_model.keras'  # Define la ruta al modelo preentrenado
model_path = r'C:...\ReconocimientoEmociones(CNN)\models\emotion_recognition_Personal.keras'
model = load_model(model_path)  # Carga el modelo preentrenado desde la ruta especificada

emotion_labels = ['happy', 'neutral', 'surprise', 'angry']  # Define las etiquetas de las emociones que el modelo puede reconocer

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