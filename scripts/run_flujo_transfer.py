# -*- coding: utf-8 -*-
"""
Ejecuta el flujo completo con Transfer Learning (para máxima precisión):
  data_split → data_preprocessing → train_transfer_imagenet.

Requisito: carpetas de emociones (angry, happy, neutral, surprise) en
data/kaggle_fer/ y en config.py: USE_KAGGLE_FER = True.

Uso (desde la raíz del proyecto):
  python scripts/run_flujo_transfer.py

En Colab: entrena con GPU, descarga el .keras y en tu laptop ejecuta
realtime_emotion_recognition.py con use_custom_cnn = False y
TRANSFER_MODEL_NAME = "EfficientNetB0" (o la base que uses).
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(script, desc):
    print("\n" + "=" * 60)
    print(f"  {desc}")
    print("=" * 60)
    ret = subprocess.run(
        [sys.executable, f"scripts/{script}"],
        cwd=str(PROJECT_ROOT),
    )
    if ret.returncode != 0:
        print(f"Error en: {desc}")
        sys.exit(ret.returncode)


def main():
    run("data_split.py", "1/3 Dividir datos (kaggle_fer → train/validation/test)")
    run("data_preprocessing.py", "2/3 Validar preprocesado")
    run("train_transfer_imagenet.py", "3/3 Entrenar con Transfer Learning (EfficientNetB0, etc.)")

    print("\n" + "=" * 60)
    print("  Listo. Para probar en vivo en tu laptop:")
    print("    1. Descarga el modelo desde Colab (models/emotion_recognition_EfficientNetB0_model.keras)")
    print("    2. Ponlo en tu proyecto local en models/")
    print("    3. En realtime_emotion_recognition.py: use_custom_cnn = False, TRANSFER_MODEL_NAME = 'EfficientNetB0'")
    print("    4. python scripts/realtime_emotion_recognition.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
