# -*- coding: utf-8 -*-
"""
Camino 6 — Transfer: base = modelo del camino 1, datos = FER.
Salida: models/modelo_camino_6.keras

Antes: camino 1 entrenado (modelo_camino_1.keras).
       USE_KAGGLE_DATABASE = True en split/preprocess, split → preprocessing.
"""
import sys
from pathlib import Path

# ========== CONFIGURA ==========
USE_KAGGLE_DATABASE = True
# ===============================

from training_utils import (  # noqa: E402
    MODEL_CAMINO_1,
    MODEL_CAMINO_6,
    ensure_kaggle_flag,
    print_training_prerequisites_banner,
    run_transfer_from_existing_model_training,
    tag_model_copy,
)

MODELO_BASE = MODEL_CAMINO_1
# Otra copia: MODELO_BASE = Path(__file__).resolve().parent.parent / "models" / "archivo.keras"


def main():
    print_training_prerequisites_banner(transfer_efficientnet=False)
    ensure_kaggle_flag(True, "Camino 6", USE_KAGGLE_DATABASE)

    base = Path(MODELO_BASE)
    if not base.is_file():
        print(f"No existe el modelo base (camino 1): {base}")
        print("Entrena primero el camino 1 o asigna MODELO_BASE a otro .keras.")
        sys.exit(1)

    print("Camino 6: transfer  base =", base.name, " → salida =", MODEL_CAMINO_6.name)
    path, acc = run_transfer_from_existing_model_training(base, MODEL_CAMINO_6)
    if path.exists():
        tag_model_copy(path, "modelo_camino_6", acc)
    print("Listo: camino 6.")


if __name__ == "__main__":
    main()
