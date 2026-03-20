# -*- coding: utf-8 -*-
"""
Camino 1 — CNN desde cero (tus fotos). Guarda en models/modelo_camino_1.keras

Antes: data_split.py → data_preprocessing.py
"""
# ========== CONFIGURA (igual que data_split.py y data_preprocessing.py) ==========
USE_KAGGLE_DATABASE = False
# =================================================================================

from training_utils import (
    MODEL_CAMINO_1,
    ensure_kaggle_flag,
    print_training_prerequisites_banner,
    run_cnn_from_scratch_training,
    tag_model_copy,
)


def main():
    print_training_prerequisites_banner(transfer_efficientnet=False)
    ensure_kaggle_flag(False, "Camino 1", USE_KAGGLE_DATABASE)
    print("Camino 1: CNN desde cero →", MODEL_CAMINO_1.name)
    path, acc = run_cnn_from_scratch_training(MODEL_CAMINO_1)
    if path.exists():
        tag_model_copy(path, "modelo_camino_1", acc)
    print("Listo: camino 1.")


if __name__ == "__main__":
    main()
