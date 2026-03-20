# -*- coding: utf-8 -*-
"""
Reconocimiento de emociones en tiempo real con la cámara.
CNN desde cero o transfer EfficientNetB0: configura rutas abajo o usa --model-path.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quiet_console

quiet_console.init()

import cv2
import numpy as np

quiet_console.silence_opencv()

from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras import Sequential  # type: ignore
from tensorflow.keras.layers import Dense as TFDense  # type: ignore

from training_utils import efficientnet_preprocess_fn  # noqa: E402
from project_config import DATA_SOURCE  # noqa: E402

quiet_console.silence_tensorflow_post_import()

# =============================================================================
# CONFIGURA AQUÍ
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
EMOTION_LIST = ["angry", "happy", "neutral", "surprise"]
FACE_SIZE = (224, 224)

# Fuente usada para entrenar el modelo actual:
# - "my_images": datos propios capturados con OpenCV (guardados en BGR)
# - "fer_2013": FER_2013 (Kaggle)
# - "affectnet": AffectNet (Kaggle)
TRAINED_DATA_SOURCE = DATA_SOURCE  # "fer_2013", "affectnet" o "my_images"

# El usuario solo cambia TRAINED_DATA_SOURCE y el script deduce el preprocesado.
if TRAINED_DATA_SOURCE == "fer_2013":
    # FER_2013: ROI en gris (luego se convierte a RGB para el modelo).
    USE_GRAY_ROI = True
    IMAGES_ARE_BGR = False
elif TRAINED_DATA_SOURCE == "affectnet":
    # AffectNet: ROI en color (flujo RGB).
    USE_GRAY_ROI = False
    IMAGES_ARE_BGR = False
elif TRAINED_DATA_SOURCE == "my_images":
    # my_images: ROI en color capturado con OpenCV (BGR).
    USE_GRAY_ROI = False
    IMAGES_ARE_BGR = True
else:
    raise ValueError(
        "TRAINED_DATA_SOURCE inválido. Usa: 'fer_2013', 'affectnet' o 'my_images'."
    )

# Ruta al .keras (relativa al proyecto o absoluta). Si vacío, usa lista por índice.
# Vacío = usa INDICE_MODELO con CANDIDATOS_EN_MODELS. O pon ruta fija, ej. "models/modelo_camino_2.keras"
MODELO_REALTIME = "models/modelo_camino_4.keras"
# Si True y no pasas --model-path, muestra selector interactivo en terminal
# con archivos exactos modelo_camino_N.keras (sin timestamp).
SELECT_MODEL_FROM_TERMINAL = True
# Si MODELO_REALTIME está vacío y no pasas --model-path, elige por índice:
CANDIDATOS_EN_MODELS = [
    "modelo_camino_1.keras",
    "modelo_camino_2.keras",
    "modelo_camino_3.keras",
    "modelo_camino_4.keras",
    "modelo_camino_5.keras",
    "modelo_camino_6.keras",
]
INDICE_MODELO = 0
# =============================================================================

_rt_parser = argparse.ArgumentParser(description="Reconocimiento de emociones en tiempo real (webcam).")
_rt_parser.add_argument(
    "--model-path",
    default=None,
    help="Ruta al .keras (prioridad sobre MODELO_REALTIME en el script).",
)
_rt_parser.add_argument(
    "--camera",
    type=int,
    default=None,
    help="Índice de cámara OpenCV.",
)
_rt_parser.add_argument(
    "--no-swap-camera",
    action="store_true",
    help="No intercambiar 0<->1 (anula SWAP_CAMERA_INDICES_0_AND_1).",
)
_rt_parser.add_argument(
    "--cli-help",
    action="store_true",
    help="Ayuda y salir.",
)
cli, _cli_unknown = _rt_parser.parse_known_args()
if cli.cli_help:
    _rt_parser.print_help()
    sys.exit(0)

MIN_CONFIDENCE = 0.50
SWAP_CAMERA_INDICES_0_AND_1 = True
SMOOTHING_ALPHA = 0.20
PREDICTION_UPDATE_EVERY_N_FRAMES = 3
BLUR_BACKGROUND = True
RT_WINDOW_TITLE = "Reconocimiento de Emociones"
# Primeros frames: intentar traer la ventana al frente (OpenCV a veces abre detrás).
WINDOW_FOCUS_ATTEMPTS = 25


def _bring_opencv_window_front(win_name: str) -> None:
    try:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 0)
    except Exception:
        pass


class DenseIgnoreQuantizationConfig(TFDense):
    """
    Compatibilidad al cargar modelos .keras guardados con Keras mas nueva.

    En algunos entornos (por ejemplo Python 3.10 / Keras 3.10-3.12),
    `Dense` no acepta el keyword `quantization_config` que aparece dentro
    del config serializado del modelo. Si lo ignoramos, `load_model` puede
    terminar y los pesos se cargan normalmente.
    """

    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)


def _list_standard_model_files() -> list[Path]:
    pattern = re.compile(r"^modelo_camino_(\d+)\.keras$")
    found: list[tuple[int, Path]] = []
    for p in MODELS_DIR.glob("*.keras"):
        m = pattern.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort(key=lambda x: x[0])
    return [p for _, p in found]


def _pick_model_from_terminal() -> Path | None:
    models = _list_standard_model_files()
    if not models:
        print("No hay modelos estándar en models/ (modelo_camino_N.keras).")
        return None

    print("\nModelos disponibles (sin timestamp):")
    for i, p in enumerate(models, start=1):
        print(f"  {i}) {p.name}")

    while True:
        raw = input("Selecciona modelo por número [Enter = cancelar]: ").strip()
        if raw == "":
            return None
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(models):
                return models[idx - 1]
        print("Opción inválida.")


def _resolve_model_path() -> Path:
    if cli.model_path:
        p = Path(cli.model_path)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    if SELECT_MODEL_FROM_TERMINAL:
        picked = _pick_model_from_terminal()
        if picked is not None:
            return picked

    ruta = (MODELO_REALTIME or "").strip()
    if ruta:
        p = Path(ruta)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    if INDICE_MODELO is not None and 0 <= INDICE_MODELO < len(CANDIDATOS_EN_MODELS):
        return MODELS_DIR / CANDIDATOS_EN_MODELS[INDICE_MODELO]

    print("Configura MODELO_REALTIME o INDICE_MODELO en este script, o usa --model-path.")
    sys.exit(1)


model_path = _resolve_model_path()

if not model_path.exists():
    print(f"Error: No se encontró el modelo en {model_path}")
    print("Entrena con generate_model_path_1..6 o edita MODELO_REALTIME / INDICE_MODELO en este script.")
    sys.exit(1)

# Carga: EfficientNet usa Lambda preprocess_fn; CNN personal suele cargar sin custom_objects.
try:
    model = load_model(str(model_path), safe_mode=False)
except Exception:
    model = load_model(
        str(model_path),
        safe_mode=False,
        # Keras guarda el nombre real de la función dentro de la capa Lambda
        # (en este proyecto: `efficientnet_preprocess_fn`), así que la clave
        # de `custom_objects` debe coincidir con ese nombre.
        custom_objects={
            "efficientnet_preprocess_fn": efficientnet_preprocess_fn,
            # Compatibilidad por si alguna versión guardó con otro alias.
            "preprocess_fn": efficientnet_preprocess_fn,
                # Compatibilidad para modelos con config que incluye
                # `quantization_config` en Dense.
                "Dense": DenseIgnoreQuantizationConfig,
        },
    )

legacy_preprocess = False
try:
    _ = model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)
except NameError as e:
    if "preprocess_fn" in str(e):
        print("Compatibilidad modelo legacy (Lambda/preprocess_fn)...")
        model = Sequential(model.layers[1:])
        legacy_preprocess = True
    else:
        raise

# Tras cargar el grafo / warm-up, TensorFlow a veces vuelve a subir verbosidad.
quiet_console.silence_tensorflow_post_import()

print(f"Modelo cargado: {model_path.name} (transfer EfficientNet usa preprocess en grafo o legacy abajo)")
_model_match = re.match(r"^modelo_camino_(\d+)\.keras$", model_path.name)
if _model_match:
    RUNNING_MODEL_LABEL = f"Corriendo: modelo camino {_model_match.group(1)}"
else:
    RUNNING_MODEL_LABEL = f"Corriendo: {model_path.name}"

emotion_labels = EMOTION_LIST

_prompt = (
    "Seleccione cámara a usar: 0 = interna, 1 = externa = "
)
if cli.camera is not None:
    camera_index = cli.camera
else:
    camera_index = int(input(_prompt))
_do_swap = SWAP_CAMERA_INDICES_0_AND_1 and not cli.no_swap_camera
if _do_swap and camera_index in (0, 1):
    camera_index = 1 - camera_index
print(f"Abriendo cámara con índice OpenCV: {camera_index} (DirectShow)")
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"Error: No se pudo abrir la cámara {camera_index}")
    sys.exit(1)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cv2.namedWindow(RT_WINDOW_TITLE, cv2.WINDOW_NORMAL)
smoothed_probs = None
cached_probs = None
frame_index = 0
window_focus_tick = 0


def draw_emotion_panel(frame, probs, labels):
    if probs is None:
        return
    h, w = frame.shape[:2]
    margin = 10
    x0 = margin
    panel_w = max(200, w - (2 * margin))
    panel_h = 26 + (len(labels) * 20) + 10
    y0 = max(0, h - panel_h - margin)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    alpha = 0.32
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(
        frame,
        "Probabilidad por emocion",
        (x0 + 10, y0 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
    )
    bar_x = x0 + 10
    bar_w = panel_w - 20
    row_h = 20
    start_y = y0 + 28
    for i, label in enumerate(labels):
        p = float(probs[i])
        pct = max(0.0, min(100.0, p * 100.0))
        y = start_y + i * row_h
        cv2.putText(
            frame,
            f"{label:8s} {pct:5.1f}%",
            (bar_x, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (240, 240, 240),
            1,
        )
        cv2.rectangle(frame, (bar_x, y + 12), (bar_x + bar_w, y + 18), (70, 70, 70), -1)
        fill = int(bar_w * p)
        color = (60, 200, 100) if i == int(np.argmax(probs)) else (80, 150, 240)
        cv2.rectangle(frame, (bar_x, y + 12), (bar_x + fill, y + 18), color, -1)
        cv2.rectangle(frame, (bar_x, y + 12), (bar_x + bar_w, y + 18), (180, 180, 180), 1)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error al leer el frame de la cámara.")
        break

    frame_index += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    display_frame = frame.copy()
    if BLUR_BACKGROUND and len(faces) > 0:
        blurred = cv2.GaussianBlur(frame, (41, 41), 0)
        display_frame = cv2.convertScaleAbs(blurred, alpha=0.72, beta=0)
        for (x, y, w, h) in faces:
            pad = int(0.10 * max(w, h))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)
            display_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

    panel_probs = None
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        if USE_GRAY_ROI:
            roi = gray[y : y + h, x : x + w]
            roi = cv2.resize(roi, FACE_SIZE, interpolation=cv2.INTER_CUBIC)
            # FER_2013 usa `color_mode="rgb"` en el preprocesado del training,
            # así que el ROI gris debe convertirse a RGB para que coincida.
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        else:
            roi = frame[y : y + h, x : x + w]
            roi = cv2.resize(roi, FACE_SIZE, interpolation=cv2.INTER_CUBIC)

        roi = roi.astype("float32") / 255.0
        if IMAGES_ARE_BGR:
            roi = roi[..., ::-1].copy()
        roi = np.expand_dims(roi, axis=0)

        try:
            run_inference = (frame_index % PREDICTION_UPDATE_EVERY_N_FRAMES == 0) or (cached_probs is None)
            if run_inference:
                if legacy_preprocess:
                    roi_for_model = efficientnet_preprocess_fn(roi * 255.0)
                else:
                    roi_for_model = roi
                prediction = model.predict(roi_for_model, verbose=0)
                raw_probs = prediction[0]
                if smoothed_probs is None:
                    smoothed_probs = raw_probs
                else:
                    smoothed_probs = (1.0 - SMOOTHING_ALPHA) * smoothed_probs + SMOOTHING_ALPHA * raw_probs
                cached_probs = smoothed_probs

            probs = cached_probs if cached_probs is not None else np.zeros(len(emotion_labels), dtype=np.float32)
            emotion_index = int(np.argmax(probs))
            emotion = emotion_labels[emotion_index]
            confidence = float(probs[emotion_index])

            if confidence >= MIN_CONFIDENCE:
                label_text = emotion
                label_color = (0, 255, 0)
            else:
                label_text = "incierto"
                label_color = (0, 165, 255)

            cv2.putText(display_frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, label_color, 2)
            panel_probs = probs
        except Exception as e:
            print(f"Error durante la predicción: {e}")

        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        smoothed_probs = None
        cached_probs = None

    draw_emotion_panel(display_frame, panel_probs, emotion_labels)
    cv2.putText(
        display_frame,
        RUNNING_MODEL_LABEL,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (225, 105, 65),  
        2,
    )
    cv2.imshow(RT_WINDOW_TITLE, display_frame)
    if window_focus_tick < WINDOW_FOCUS_ATTEMPTS:
        _bring_opencv_window_front(RT_WINDOW_TITLE)
        window_focus_tick += 1

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or cv2.getWindowProperty(RT_WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
