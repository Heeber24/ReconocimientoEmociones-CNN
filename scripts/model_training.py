# -*- coding: utf-8 -*-
"""
Entrenamiento por Transfer Learning: base preentrenada (ImageNet) + cabeza para 4 emociones.
Usa generadores de data_preprocessing; no define preprocesado. Paso 4 del flujo (una de las dos opciones).
Flujo y modelos disponibles: ver README y NOTAS_TECNICAS en la raíz.
"""
# Base preentrenada: EfficientNetB0 (por defecto), VGG16, ResNet50, MobileNetV2, DenseNet121.
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

# Mapeo: nombre -> (clase del modelo, input_shape). Todos 224x224 para este proyecto.
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

# Verificar directorios (usan config.DATA_ROOT)
for path in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:
    if not os.path.exists(path):
        print(f"Error: El directorio no existe: {path}")
        print("Ejecuta antes data_split.py o apunta config.DATA_ROOT a tu dataset (p. ej. repo).")
        exit(1)

# Generadores desde data_preprocessing (único lugar con preprocesado y augmentation)
train_generator = get_train_generator(
    target_size=INPUT_SHAPE[:2], batch_size=32, seed=42
)
validation_generator = get_validation_generator(
    target_size=INPUT_SHAPE[:2], batch_size=32, seed=42
)
test_generator = get_test_generator(
    target_size=INPUT_SHAPE[:2], batch_size=32, shuffle=False, seed=42
)

# Modelo base preentrenado (elige BASE_MODEL_NAME al inicio del script)
base_model = BaseModelClass(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
print(f"Usando base: {BASE_MODEL_NAME}")

# Congelar capas base
base_model.trainable = False  # Congela las capas base del modelo para preservar el conocimiento preentrenado

# Modelo secuencial
model = Sequential([  # Crea un modelo secuencial
    base_model,  # Agrega el modelo base como capa inicial
    GlobalAveragePooling2D(),  # Agrega una capa de promedio global
    Dense(1024, activation='relu'),  # Agrega una capa densa con activación ReLU
    BatchNormalization(),  # Agrega una capa de normalización por lotes
    Dropout(0.5),  # Agrega una capa de dropout para regularización
    Dense(train_generator.num_classes, activation='softmax')  # Agrega la capa de salida con activación softmax
])

# Compilación del modelo
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),  # Compila el modelo con el optimizador Adam y tasa de aprendizaje 1e-4
              loss='categorical_crossentropy',  # Función de pérdida categórica cruzada
              metrics=['accuracy'])  # Métrica de precisión

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Callback para detener el entrenamiento temprano
checkpoint = ModelCheckpoint(
    str(MODEL_TRANSFER),
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6)  # Callback para reducir la tasa de aprendizaje

# Fijar semillas
np.random.seed(42)  # Fija la semilla aleatoria de NumPy
tf.random.set_seed(42)  # Fija la semilla aleatoria de TensorFlow

# Entrenamiento inicial
model.fit(  # Entrena el modelo
    train_generator,  # Generador de datos de entrenamiento
    steps_per_epoch=train_generator.samples // train_generator.batch_size,  # Número de pasos por época
    epochs=50,  # Número de épocas
    validation_data=validation_generator,  # Generador de datos de validación
    validation_steps=validation_generator.samples // validation_generator.batch_size,  # Número de pasos de validación
    callbacks=[early_stopping, checkpoint, lr_scheduler]  # Lista de callbacks a utilizar
)


# Evaluación inicial
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

# Evaluación final
scores = model.evaluate(test_generator)
print(f"Precisión del modelo en el conjunto de prueba después del ajuste fino: {scores[1] * 100:.2f}%")