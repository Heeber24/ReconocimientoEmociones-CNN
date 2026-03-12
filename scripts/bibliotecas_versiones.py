# -*- coding: utf-8 -*-
"""
Comprueba que tengas instaladas las bibliotecas necesarias para el proyecto.
Ejecuta: pip install -r requirements.txt (desde la raíz del proyecto)
"""
import tensorflow as tf  # pip install tensorflow
import numpy as np       # pip install numpy
import cv2              # pip install opencv-python
import sklearn          # pip install scikit-learn
import shutil           # módulo estándar de Python, no requiere pip

print("Versiones instaladas:")
print(f"  TensorFlow: {tf.__version__}")
print(f"  NumPy:      {np.__version__}")
print(f"  OpenCV:     {cv2.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")
print("\nSi falta alguna, instala con: pip install -r requirements.txt")

