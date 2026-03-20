# -*- coding: utf-8 -*-
"""
Configuración global del proyecto para selección de dataset fuente.

Cambiar SOLO `DATA_SOURCE` y reutilizar en split, preprocessing y realtime.
"""
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fuente de datos activa del proyecto:
# - "fer_2013"  -> data/FER_2013
# - "affectnet" -> data/AffectNet
# - "my_images" -> data/my_images
DATA_SOURCE = "my_images"

DATASET_DIRS = {
    "fer_2013": PROJECT_ROOT / "data" / "FER_2013",
    "affectnet": PROJECT_ROOT / "data" / "AffectNet",
    "my_images": PROJECT_ROOT / "data" / "my_images",
}

PREPARED_DATA_DIR = PROJECT_ROOT / "data" / "prepared_data"

