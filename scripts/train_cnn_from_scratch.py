# -*- coding: utf-8 -*-
"""
Entrena una red CNN desde cero (sin base preentrenada).

Construye y entrena una red convolucional propia; usa los generadores de
data_preprocessing. Paso 4 del flujo; alternativa a train_transfer_imagenet.py.
Ver README para los 6 caminos y NOTAS_TECNICAS para comparar con Transfer Learning.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # type: ignore

from config import TRAIN_DIR, VALIDATION_DIR, MODEL_CUSTOM
from data_preprocessing import get_train_generator, get_validation_generator

# --- Configuración (ajustar según datos y necesidades) ---
input_shape = (224, 224, 3)
num_classes = 4
batch_size = 64
epochs = 30  # Máximo; EarlyStopping puede terminar antes si val_loss no mejora
early_stopping_patience = 10

# Verificar GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detectada: {gpus[0].name}")
else:
    print("No se detectó GPU. Usando CPU.")

def create_custom_cnn(input_shape, num_classes):
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        # Fully Connected
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

model = create_custom_cnn(input_shape, num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_generator = get_train_generator(batch_size=batch_size)
val_generator = get_validation_generator(batch_size=batch_size)

checkpoint = ModelCheckpoint(
    str(MODEL_CUSTOM),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=early_stopping_patience,
    restore_best_weights=True
)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[checkpoint, early_stopping]
)
