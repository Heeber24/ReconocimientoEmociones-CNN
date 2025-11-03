# -*- coding: utf-8 -*-
"""
Script para la captura de rostros utilizando OpenCV con un menú de selección.

Este programa detecta rostros desde una cámara en tiempo real, los procesa y
los guarda en una carpeta específica para una emoción seleccionada de un menú.
"""
from pathlib import Path
import cv2
import time

## --- CONFIGURACIÓN ---
# Se restaura la ruta de guardado exacta solicitada.
DATA_PATH = Path(r'C:...\ReconocimientoEmociones(CNN)\data\Tus_Imagenes\data_collection')

# Lista de emociones permitidas para el menú de selección.
EMOTION_LIST = ["angry", "happy", "neutral", "surprise"]

# Tamaño al que se redimensionarán los rostros capturados (ancho, alto).
FACE_SIZE = (224, 224)

# Número máximo de imágenes a capturar por cada emoción.
MAX_IMAGES = 50
# --------------------


def capture_faces():
    """
    Función principal que maneja la lógica de captura, procesamiento y guardado de rostros.
    """
    # --- 1. SELECCIÓN DE EMOCIÓN Y CÁMARA ---
    
    # --- Menú de selección de emoción ---
    print(">> Selecciona la emoción a capturar:")
    # Se crea un diccionario para mapear el número de opción a la emoción.
    emotion_map = {i + 1: emotion for i, emotion in enumerate(EMOTION_LIST)}
    for num, emotion in emotion_map.items():
        print(f"   {num}: {emotion}")

    emotion_name = None
    # Bucle para asegurar que el usuario ingrese una opción válida.
    while emotion_name is None:
        try:
            choice = int(input("Ingresa el número de tu elección: "))
            # Se verifica si la elección está en nuestro mapa de emociones.
            if choice in emotion_map:
                emotion_name = emotion_map[choice]
            else:
                print("Error: Número fuera de rango. Inténtalo de nuevo.")
        except ValueError:
            print("Error: Debes ingresar un número. Inténtalo de nuevo.")
    
    print(f"Emoción seleccionada: '{emotion_name}'")
    
    # --- Selección de cámara (se mantiene como en la versión anterior) ---
    try:
        camera_index = int(input("\n>> Seleccione la cámara (0 = interna, 1 = externa): "))
    except ValueError:
        print("Entrada inválida. Usando cámara 0 por defecto.")
        camera_index = 0

    # --- 2. CONFIGURACIÓN DE RUTAS Y ARCHIVOS ---
    emotions_path = DATA_PATH / emotion_name
    print(f"Las imágenes se guardarán en: {emotions_path}")
    emotions_path.mkdir(parents=True, exist_ok=True)

    haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_classifier = cv2.CascadeClassifier(haar_cascade_path)
    if face_classifier.empty():
        print("Error: No se pudo cargar el clasificador Haar Cascade.")
        return

    # --- 3. INICIALIZACIÓN DE LA CÁMARA ---
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara {camera_index}.")
        return

    # --- 4. BUCLE PRINCIPAL DE CAPTURA ---
    count = 0
    is_capturing = False

    print("\nTodo listo. En la ventana de la cámara, presiona 'C' para INICIAR/PAUSAR la captura.")
    print("Presiona 'ESC' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo obtener el fotograma de la cámara. Saliendo.")
            break

        frame = cv2.flip(frame, 1)
        frame_display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if is_capturing and count < MAX_IMAGES:
            faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(50, 50))
            
            if len(faces) > 0:
                # Se enfoca solo en el rostro más grande detectado.
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                (x, y, w, h) = largest_face
                
                cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                face_roi_gray = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi_gray, FACE_SIZE, interpolation=cv2.INTER_CUBIC)
                
                image_filename = emotions_path / f"rostro_{emotion_name}_{count}.png"
                cv2.imwrite(str(image_filename), face_resized)
                count += 1
                time.sleep(0.1)

        # --- 5. MOSTRAR INFORMACIÓN EN PANTALLA ---
        status_text = "CAPTURANDO" if is_capturing else "PAUSADO"
        status_color = (0, 255, 0) if is_capturing else (0, 0, 255)
        
        cv2.putText(frame_display, f"Estado: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame_display, f"Imagenes: {count}/{MAX_IMAGES}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_display, "Presiona 'C' para iniciar/pausar", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame_display, "Presiona 'ESC' para salir", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow('Captura de Rostros para Dataset', frame_display)

        # --- 6. MANEJO DE TECLADO ---
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or count >= MAX_IMAGES:
            break
        elif key == ord('c'):
            is_capturing = not is_capturing

    # --- 7. LIBERACIÓN DE RECURSOS ---
    print(f"\nCaptura finalizada. Se guardaron {count} imágenes para la emoción '{emotion_name}'.")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_faces()