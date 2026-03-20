# -*- coding: utf-8 -*-
"""
Comprueba que tengas instaladas las bibliotecas necesarias para el proyecto.

Desde la raíz del proyecto:
  python scripts/bibliotecas_versiones.py

Si algo falla, instala dependencias con:
  pip install -r requirements.txt
"""
from __future__ import annotations

import importlib
import sys


def _ok(name: str, version: str, pip_hint: str) -> None:
    print(f"  [OK] {name}: {version}")
    print(f"       ({pip_hint})")


def _fail(name: str, err: Exception, pip_hint: str) -> None:
    print(f"  [FALTA] {name}: {err}")
    print(f"          {pip_hint}")


def main() -> int:
    print("Comprobación de librerías — ReconocimientoEmociones-CNN\n")

    try:
        import tensorflow as tf

        _ok("TensorFlow", tf.__version__, "pip install tensorflow")
        _ok("Keras (tf.keras)", tf.keras.__version__, "viene con TensorFlow")
    except ImportError as e:
        _fail("TensorFlow / Keras", e, "pip install tensorflow")

    try:
        import numpy as np

        _ok("NumPy", np.__version__, "pip install numpy")
    except ImportError as e:
        _fail("NumPy", e, "pip install numpy")

    try:
        import cv2

        _ok("OpenCV (cv2)", cv2.__version__, "pip install opencv-python")
    except ImportError as e:
        _fail("OpenCV", e, "pip install opencv-python")

    try:
        import sklearn

        _ok("scikit-learn", sklearn.__version__, "pip install scikit-learn")
    except ImportError as e:
        _fail("scikit-learn", e, "pip install scikit-learn")

    try:
        from PIL import Image  # noqa: F401

        pil = importlib.import_module("PIL")
        _ok("Pillow", getattr(pil, "__version__", "?"), "pip install Pillow")
    except ImportError as e:
        _fail("Pillow", e, "pip install Pillow")

    try:
        import scipy

        _ok("SciPy", scipy.__version__, "pip install scipy")
    except ImportError as e:
        _fail("SciPy", e, "pip install scipy")

    try:
        import matplotlib

        _ok("Matplotlib", matplotlib.__version__, "pip install matplotlib")
    except ImportError as e:
        _fail("Matplotlib", e, "pip install matplotlib")

    import shutil

    print(f"  [OK] shutil: módulo estándar de Python (no se instala con pip)")

    print("\n" + "-" * 60)
    print("Listado completo de dependencias: requirements.txt")
    print("Entorno y notas: NOTAS_TECNICAS.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
