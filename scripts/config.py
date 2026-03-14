# -*- coding: utf-8 -*-
"""
Configuración central: rutas y parámetros del proyecto.
Rutas relativas a la raíz; DATA_ROOT permite apuntar a otro dataset (p. ej. repo).
Ver README para flujo y uso de DATA_ROOT.
"""
from pathlib import Path

# Raíz del proyecto (carpeta que contiene 'scripts', 'data', 'models')
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset: raw captures and train/validation/test splits (see README)
# Si usas FER2013 en data/kaggle_fer/, pon USE_KAGGLE_FER = True.
USE_KAGGLE_FER = True  # True = leer desde data/kaggle_fer para data_split
DATA_COLLECTION = (PROJECT_ROOT / "data" / "kaggle_fer") if USE_KAGGLE_FER else (PROJECT_ROOT / "data" / "images" / "data_collection")
PREPARED_DATA = PROJECT_ROOT / "data" / "images" / "prepared_data"

# Raíz de datos para entrenamiento. Por defecto = prepared_data del proyecto.
# Para usar un repo externo o otra ruta, cambia DATA_ROOT (misma estructura:
# train/, validation/, test/, cada uno con subcarpetas por emoción).
DATA_ROOT = PREPARED_DATA
TRAIN_DIR = DATA_ROOT / "train"
VALIDATION_DIR = DATA_ROOT / "validation"
TEST_DIR = DATA_ROOT / "test"

# Modelos guardados
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_TRANSFER = MODELS_DIR / "emotion_recognition_EfficientNetB0_model.keras"
MODEL_CUSTOM = MODELS_DIR / "emotion_recognition_Personal.keras"
# Resultado de transfer desde tu modelo (train_transfer_from_my_model.py)
MODEL_FROM_CUSTOM = MODELS_DIR / "emotion_recognition_from_custom.keras"

# Emociones (orden alfabético: debe coincidir con flow_from_directory)
EMOTION_LIST = ["angry", "happy", "neutral", "surprise"]

# Parámetros de imagen (deben ser coherentes en captura, preprocesado y modelo)
FACE_SIZE = (224, 224)
IMG_SHAPE = (224, 224, 3)
# True si las imágenes vienen de data_collection.py (OpenCV guarda BGR). False si usas FER2013 (kaggle_fer).
IMAGES_ARE_BGR = False  # False para FER2013 en kaggle_fer
