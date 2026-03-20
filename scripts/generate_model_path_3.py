# -*- coding: utf-8 -*-
"""
Camino 3 — CNN desde cero (FER). Guarda en models/modelo_camino_3.keras

Antes: data_split.py → data_preprocessing.py
"""
# ========== CONFIGURA (igual que data_split.py y data_preprocessing.py) ==========
USE_KAGGLE_DATABASE = True
# =================================================================================

from training_utils import (
    MODEL_CAMINO_3,
    ensure_kaggle_flag,
    print_training_prerequisites_banner,
    run_cnn_from_scratch_training,
    tag_model_copy,
)


def main():
    print_training_prerequisites_banner(transfer_efficientnet=False)
    ensure_kaggle_flag(True, "Camino 3", USE_KAGGLE_DATABASE)
    print("Camino 3: CNN FER →", MODEL_CAMINO_3.name)
    path, acc = run_cnn_from_scratch_training(MODEL_CAMINO_3)
    if path.exists():
        tag_model_copy(path, "modelo_camino_3", acc)
    print("Listo: camino 3.")


if __name__ == "__main__":
    main()
