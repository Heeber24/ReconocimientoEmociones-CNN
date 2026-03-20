# -*- coding: utf-8 -*-
"""
Camino 4 — Transfer EfficientNetB0 (FER). Guarda en models/modelo_camino_4.keras

Antes: data_split.py → data_preprocessing.py
"""
# ========== CONFIGURA (igual que data_split.py y data_preprocessing.py) ==========
USE_KAGGLE_DATABASE = True
# =================================================================================

from training_utils import (
    MODEL_CAMINO_4,
    ensure_kaggle_flag,
    print_training_prerequisites_banner,
    run_transfer_efficientnet_training,
    tag_model_copy,
)


def main():
    print_training_prerequisites_banner(transfer_efficientnet=True)
    ensure_kaggle_flag(True, "Camino 4", USE_KAGGLE_DATABASE)
    print("Camino 4: EfficientNet FER →", MODEL_CAMINO_4.name)
    path, acc = run_transfer_efficientnet_training(MODEL_CAMINO_4)
    if path.exists():
        tag_model_copy(path, "modelo_camino_4", acc)
    print("Listo: camino 4.")


if __name__ == "__main__":
    main()
