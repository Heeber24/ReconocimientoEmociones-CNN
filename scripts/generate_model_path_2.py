# -*- coding: utf-8 -*-
"""
Camino 2 — Transfer EfficientNetB0 (tus fotos). Guarda en models/modelo_camino_2.keras

Antes: data_split.py → data_preprocessing.py
"""
# ========== CONFIGURA (igual que data_split.py y data_preprocessing.py) ==========
USE_KAGGLE_DATABASE = False
# =================================================================================

from training_utils import (
    MODEL_CAMINO_2,
    ensure_kaggle_flag,
    print_training_prerequisites_banner,
    run_transfer_efficientnet_training,
    tag_model_copy,
)


def main():
    print_training_prerequisites_banner(transfer_efficientnet=True)
    ensure_kaggle_flag(False, "Camino 2", USE_KAGGLE_DATABASE)
    print("Camino 2: EfficientNet →", MODEL_CAMINO_2.name)
    path, acc = run_transfer_efficientnet_training(MODEL_CAMINO_2)
    if path.exists():
        tag_model_copy(path, "modelo_camino_2", acc)
    print("Listo: camino 2.")


if __name__ == "__main__":
    main()
