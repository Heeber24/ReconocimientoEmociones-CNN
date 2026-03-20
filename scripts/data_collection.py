# -*- coding: utf-8 -*-
"""
Captura de rostros por webcam para el dataset de emociones.
Detecta rostros en tiempo real, los redimensiona a 224x224 en color (BGR al guardar;
el preprocesado convierte a RGB para entrenar). Paso 1 del flujo.
Flujo completo y opciones: ver README en la raíz del proyecto.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quiet_console

quiet_console.init()

import cv2
import time

quiet_console.silence_opencv()

# =============================================================================
# CONFIGURA AQUÍ — solo captura para datos propios (carpeta destino)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "my_images"
EMOTION_LIST = ["angry", "happy", "neutral", "surprise"]
FACE_SIZE = (224, 224)
# =============================================================================

# Número máximo de imágenes a capturar por cada emoción.
MAX_IMAGES = 400
# Mantener la misma lógica que realtime_emotion_recognition.py:
# si seleccionas 0/1, se intercambia para abrir el índice OpenCV correcto.
SWAP_CAMERA_INDICES_0_AND_1 = True
# Misma vista que realtime: fondo difuminado y rostro nítido (solo visual; no altera lo guardado).
BLUR_BACKGROUND = True
WINDOW_TITLE = "Captura de Rostros para Dataset"
# Primeros frames: intentar traer la ventana al frente (Windows / OpenCV a veces abre detrás).
WINDOW_FOCUS_ATTEMPTS = 25
# Mismo tono que realtime_emotion_recognition (etiqueta del modelo) — BGR.
UI_TEXT_ROYAL_BLUE_BGR = (225, 105, 65)
# --------------------


def _bring_opencv_window_front(win_name: str) -> None:
    """Pone la ventana encima del resto (si el backend de OpenCV lo permite)."""
    try:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 0)
    except Exception:
        pass


def _user_confirms_stop_capture() -> bool:
    """True = sí salir. Solo se usa cuando el usuario cerró la ventana estando en captura."""
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError:
        return True
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass
    root.update()
    ok = messagebox.askyesno(
        "Captura activa",
        "¿Seguro que quieres dejar de capturar?",
        icon="warning",
        parent=root,
    )
    root.destroy()
    return ok


def _draw_capture_zone_explicit(frame, x, y, w, h) -> None:
    """Marco muy visible: es exactamente el rectángulo que se recorta y guarda."""
    fh, fw = frame.shape[:2]
    cv2.rectangle(frame, (x - 3, y - 3), (x + w + 2, y + h + 2), (255, 255, 255), 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
    label = "SE GUARDA ESTE ROSTRO (224x224)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.55, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    pad = 6
    if y >= th + 2 * pad + 8:
        y0, y1 = y - th - 2 * pad, y
    else:
        y0, y1 = y + h + 6, min(fh - 2, y + h + th + 2 * pad + 6)
    x1 = min(fw - 1, x + tw + 2 * pad)
    cv2.rectangle(frame, (max(0, x), max(0, y0)), (x1, min(fh - 1, y1)), (0, 100, 0), -1)
    cv2.putText(
        frame,
        label,
        (max(0, x) + pad,
         min(fh - 4, y1 - pad)),
        font,
        scale,
        (255, 255, 255),
        thick,
    )


def _draw_capture_preview(frame, x, y, w, h) -> None:
    """Guía cuando aún no pulsaste C (no se guarda nada)."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 100), 2)
    label = "Zona de Captura"
    ty = max(22, y - 8)
    cv2.putText(frame, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 255, 200), 2)


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
    
    # --- Selección de cámara (misma lógica de realtime_emotion_recognition.py) ---
    try:
        camera_index = int(input("\n>> Seleccione la cámara (0 = interna, 1 = externa): "))
    except ValueError:
        print("Entrada inválida. Usando cámara 0 por defecto.")
        camera_index = 0
    if SWAP_CAMERA_INDICES_0_AND_1 and camera_index in (0, 1):
        camera_index = 1 - camera_index
    print(f"Abriendo cámara con índice OpenCV: {camera_index} (DirectShow)")

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

    print("\nTodo listo. 'C' = iniciar/pausar captura. Cierra con la X de la ventana (como en realtime).")

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    window_focus_tick = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo obtener el fotograma de la cámara. Saliendo.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(50, 50))

        frame_display = frame.copy()
        if BLUR_BACKGROUND and len(faces) > 0:
            blurred = cv2.GaussianBlur(frame, (41, 41), 0)
            frame_display = cv2.convertScaleAbs(blurred, alpha=0.72, beta=0)
            for (fx, fy, fw, fh) in faces:
                pad = int(0.10 * max(fw, fh))
                x1 = max(0, fx - pad)
                y1 = max(0, fy - pad)
                x2 = min(frame.shape[1], fx + fw + pad)
                y2 = min(frame.shape[0], fy + fh + pad)
                frame_display[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

        largest_face = None
        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])

        if is_capturing and count < MAX_IMAGES and largest_face is not None:
            x, y, w, h = largest_face
            # Recorte siempre desde `frame` (nítido), no desde frame_display.
            face_roi_color = frame[y : y + h, x : x + w]
            face_resized = cv2.resize(face_roi_color, FACE_SIZE, interpolation=cv2.INTER_CUBIC)
            image_filename = emotions_path / f"rostro_{emotion_name}_{count}.png"
            cv2.imwrite(str(image_filename), face_resized)
            count += 1
            time.sleep(0.1)

        if largest_face is not None:
            lx, ly, lw, lh = largest_face
            if is_capturing:
                _draw_capture_zone_explicit(frame_display, lx, ly, lw, lh)
            else:
                _draw_capture_preview(frame_display, lx, ly, lw, lh)

        # --- 5. MOSTRAR INFORMACIÓN EN PANTALLA ---
        status_text = "CAPTURANDO" if is_capturing else "PAUSADO"
        status_color = (0, 255, 0) if is_capturing else (0, 0, 255)
        
        cv2.putText(frame_display, f"Estado: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame_display, f"Imagenes: {count}/{MAX_IMAGES}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(
            frame_display,
            "Presiona 'C' para iniciar/pausar",
            (10, frame.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            UI_TEXT_ROYAL_BLUE_BGR,
            2,
        )

        cv2.imshow(WINDOW_TITLE, frame_display)
        if window_focus_tick < WINDOW_FOCUS_ATTEMPTS:
            _bring_opencv_window_front(WINDOW_TITLE)
            window_focus_tick += 1

        # --- 6. MANEJO DE TECLADO Y CIERRE CON LA X (igual idea que realtime) ---
        key = cv2.waitKey(1) & 0xFF

        if count >= MAX_IMAGES:
            break
        elif key == ord("c"):
            is_capturing = not is_capturing
        else:
            try:
                win_gone = cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1
            except Exception:
                win_gone = True
            if win_gone:
                if is_capturing:
                    if _user_confirms_stop_capture():
                        break
                    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
                    window_focus_tick = 0
                else:
                    break

    # --- 7. LIBERACIÓN DE RECURSOS ---
    print(f"\nCaptura finalizada. Se guardaron {count} imágenes para la emoción '{emotion_name}'.")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_faces()