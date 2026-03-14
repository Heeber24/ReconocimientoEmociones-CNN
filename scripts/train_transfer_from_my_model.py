# -*- coding: utf-8 -*-
"""
Entrena una nueva cabeza usando TU modelo guardado como base.

Cargas un modelo que ya entrenaste (p. ej. train_cnn_from_scratch o
train_transfer_imagenet), quitas la capa de salida, congelas el resto
y entrenas una nueva cabeza. El resultado se guarda en models/ y puedes
usarlo en realtime_emotion_recognition como los demás.

Flujo: train_cnn_from_scratch.py → guarda Personal.keras → este script
       lo carga como base → entrena nueva cabeza → guarda emotion_recognition_from_custom.keras
Ver README, caminos 5 y 6.
"""
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # type: ignore

from config import TRAIN_DIR, VALIDATION_DIR, TEST_DIR, MODELS_DIR, MODEL_CUSTOM, MODEL_FROM_CUSTOM
from data_preprocessing import get_train_generator, get_validation_generator, get_test_generator

# ¿Qué modelo cargar como base? Por defecto el de train_cnn_from_scratch (Personal.keras).
# Puedes poner otro .keras que tengas en models/ (p. ej. uno de train_transfer_imagenet).
MODEL_AS_BASE = MODEL_CUSTOM

def main():
    if not MODEL_AS_BASE.exists():
        print(f"Error: No existe el modelo base: {MODEL_AS_BASE}")
        print("Entrena antes train_cnn_from_scratch.py (o el modelo que quieras usar como base).")
        sys.exit(1)

    for path in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:
        if not os.path.exists(path):
            print(f"Error: No existe {path}. Ejecuta data_split y data_preprocessing.")
            sys.exit(1)

    print("Cargando modelo como base:", MODEL_AS_BASE)
    model = load_model(str(MODEL_AS_BASE))

    model.pop()
    for layer in model.layers:
        layer.trainable = False

    train_gen = get_train_generator(target_size=(224, 224), batch_size=32, seed=42)
    val_gen = get_validation_generator(target_size=(224, 224), batch_size=32, seed=42)
    test_gen = get_test_generator(target_size=(224, 224), batch_size=32, shuffle=False, seed=42)

    model.add(Dense(train_gen.num_classes, activation='softmax'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Entrenando nueva cabeza (base congelada). El modelo resultante se guardará en models/.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        str(MODEL_FROM_CUSTOM),
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6)

    np.random.seed(42)
    tf.random.set_seed(42)

    model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        epochs=50,
        validation_data=val_gen,
        validation_steps=val_gen.samples // val_gen.batch_size,
        callbacks=[checkpoint, early, lr_reduce]
    )

    scores = model.evaluate(test_gen)
    print(f"Precisión en test: {scores[1] * 100:.2f}%")
    print(f"Modelo guardado en: {MODEL_FROM_CUSTOM}")
    print("Para usarlo en tiempo real, en realtime_emotion_recognition.py pon use_from_custom = True.")

if __name__ == "__main__":
    main()
