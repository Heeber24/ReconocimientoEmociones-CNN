# -*- coding: utf-8 -*-
"""
Entrena con Transfer Learning usando una base preentrenada en ImageNet.

Elige una base (EfficientNetB0, VGG16, ResNet50, MobileNetV2, DenseNet121),
añade una cabeza para 4 emociones y entrena. Usa generadores de data_preprocessing.
Paso 4 del flujo; alternativa a train_cnn_from_scratch.py. Ver README y NOTAS_TECNICAS.
"""
# Base preentrenada: cambia BASE_MODEL_NAME para usar otra.
BASE_MODEL_NAME = "EfficientNetB0"

import tensorflow as tf
from tensorflow.keras.applications import (  # type: ignore
    EfficientNetB0,
    VGG16,
    ResNet50,
    MobileNetV2,
    DenseNet121,
)
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # type: ignore
import os
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TRAIN_DIR, VALIDATION_DIR, TEST_DIR, MODELS_DIR
from data_preprocessing import get_train_generator, get_validation_generator, get_test_generator

AVAILABLE_BASE_MODELS = {
    "EfficientNetB0": (EfficientNetB0, (224, 224, 3)),
    "VGG16": (VGG16, (224, 224, 3)),
    "ResNet50": (ResNet50, (224, 224, 3)),
    "MobileNetV2": (MobileNetV2, (224, 224, 3)),
    "DenseNet121": (DenseNet121, (224, 224, 3)),
}
BaseModelClass, INPUT_SHAPE = AVAILABLE_BASE_MODELS.get(
    BASE_MODEL_NAME, AVAILABLE_BASE_MODELS["EfficientNetB0"]
)
MODEL_TRANSFER = MODELS_DIR / f"emotion_recognition_{BASE_MODEL_NAME}_model.keras"

for path in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:
    if not os.path.exists(path):
        print(f"Error: El directorio no existe: {path}")
        print("Ejecuta antes data_split.py o apunta config.DATA_ROOT a tu dataset (p. ej. repo).")
        exit(1)

train_generator = get_train_generator(
    target_size=INPUT_SHAPE[:2], batch_size=32, seed=42
)
validation_generator = get_validation_generator(
    target_size=INPUT_SHAPE[:2], batch_size=32, seed=42
)
test_generator = get_test_generator(
    target_size=INPUT_SHAPE[:2], batch_size=32, shuffle=False, seed=42
)

base_model = BaseModelClass(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
print(f"Usando base: {BASE_MODEL_NAME}")

base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    str(MODEL_TRANSFER),
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6)

np.random.seed(42)
tf.random.set_seed(42)

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping, checkpoint, lr_scheduler]
)

scores = model.evaluate(test_generator)
print(f"Precisión del modelo en el conjunto de prueba: {scores[1] * 100:.2f}%")

# Ajuste fino
base_model.trainable = True
fine_tune_at = int(len(base_model.layers) * 0.8)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping, checkpoint, lr_scheduler]
)

scores = model.evaluate(test_generator)
print(f"Precisión después del ajuste fino: {scores[1] * 100:.2f}%")
