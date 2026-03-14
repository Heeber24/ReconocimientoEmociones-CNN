# -*- coding: utf-8 -*-
"""
Ejecuta todo el flujo: data_split → data_preprocessing → train_cnn_from_scratch.

Requisito: tener las carpetas de emociones (angry, happy, neutral, surprise) en
data/kaggle_fer/ y en config.py: USE_KAGGLE_FER = True, IMAGES_ARE_BGR = False.

Uso (desde la raíz del proyecto):
  python scripts/run_flujo_completo.py

Al terminar, ejecuta: python scripts/realtime_emotion_recognition.py (use_custom_cnn = True).
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
    run("train_cnn_from_scratch.py", "3/3 Entrenar CNN desde cero (puede tardar 15-30 min)")

    print("\n" + "=" * 60)
    print("  Listo. Para probar en vivo:")
    print("    python scripts/realtime_emotion_recognition.py")
    print("  (En realtime deja use_custom_cnn = True)")
    print("=" * 60)


if __name__ == "__main__":
    main()
