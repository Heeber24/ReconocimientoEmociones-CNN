# -*- coding: utf-8 -*-
"""
Camino 5 — Transfer: base = modelo del camino 3, datos = tus fotos.
Salida: models/modelo_camino_5.keras

Antes: camino 3 entrenado (modelo_camino_3.keras).
       USE_KAGGLE_DATABASE = False en split/preprocess, split → preprocessing.
"""
import sys
from pathlib import Path

# ========== CONFIGURA ==========
USE_KAGGLE_DATABASE = False
# ===============================

from training_utils import (  # noqa: E402
    MODEL_CAMINO_3,
    MODEL_CAMINO_5,
    ensure_kaggle_flag,
    print_training_prerequisites_banner,
    run_transfer_from_existing_model_training,
    tag_model_copy,
)

# Base del transfer: por defecto modelo_camino_3.keras (camino 3).
MODELO_BASE = MODEL_CAMINO_3
# Otra copia: MODELO_BASE = Path(__file__).resolve().parent.parent / "models" / "archivo.keras"


def main():
    print_training_prerequisites_banner(transfer_efficientnet=False)
    ensure_kaggle_flag(False, "Camino 5", USE_KAGGLE_DATABASE)

    base = Path(MODELO_BASE)
    if not base.is_file():
        print(f"No existe el modelo base (camino 3): {base}")
        print("Entrena primero el camino 3 o asigna MODELO_BASE a otro .keras.")
        sys.exit(1)

    print("Camino 5: transfer  base =", base.name, " → salida =", MODEL_CAMINO_5.name)
    path, acc = run_transfer_from_existing_model_training(base, MODEL_CAMINO_5)
    if path.exists():
        tag_model_copy(path, "modelo_camino_5", acc)
    print("Listo: camino 5.")


if __name__ == "__main__":
    main()
