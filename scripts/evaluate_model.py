# -*- coding: utf-8 -*-
"""Evalúa el modelo guardado (Personal.keras) en el conjunto de test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tensorflow.keras.models import load_model  # type: ignore
from config import MODEL_CUSTOM, TEST_DIR
from data_preprocessing import get_test_generator

if not MODEL_CUSTOM.exists():
    print("No existe", MODEL_CUSTOM)
    sys.exit(1)

print("Cargando modelo:", MODEL_CUSTOM)
model = load_model(str(MODEL_CUSTOM))
test_gen = get_test_generator(batch_size=32, shuffle=False, seed=42)

print("Evaluando en test...")
loss, accuracy = model.evaluate(test_gen)
print(f"\n--- Resultado en TEST ---")
print(f"  Loss:     {loss:.4f}")
print(f"  Accuracy: {accuracy*100:.2f}%")
