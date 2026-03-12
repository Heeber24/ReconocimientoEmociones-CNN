# -*- coding: utf-8 -*-
"""
Entrenamiento con CNN desde cero (sin Transfer Learning).
Usa generadores de data_preprocessing; no define preprocesado. Paso 4 del flujo (alternativa a model_training).
Ver README y NOTAS_TECNICAS para comparar con Transfer Learning.
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
gpus = tf.config.list_physical_devices('GPU')  # Obtiene la lista de GPUs disponibles
if gpus:  # Si hay GPUs disponibles
    print(f"GPU detectada: {gpus[0].name}")  # Imprime el nombre de la primera GPU detectada
else:  # Si no hay GPUs disponibles
    print("No se detectó GPU. Usando CPU.")  # Imprime un mensaje indicando que se usará la CPU

# Crear modelo optimizado
def create_custom_cnn(input_shape, num_classes):  # Define una función para crear el modelo CNN personalizado
    model = Sequential([  # Crea un modelo secuencial
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),  # Capa convolucional 2D con 32 filtros
        BatchNormalization(),  # Capa de normalización por lotes
        MaxPooling2D(pool_size=(2, 2)),  # Capa de max pooling para reducir la dimensionalidad
        Dropout(0.25),  # Capa de dropout para regularización

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),  # Capa convolucional 2D con 64 filtros
        BatchNormalization(),  # Capa de normalización por lotes
        MaxPooling2D(pool_size=(2, 2)),  # Capa de max pooling
        Dropout(0.25),  # Capa de dropout

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),  # Capa convolucional 2D con 128 filtros
        BatchNormalization(),  # Capa de normalización por lotes
        MaxPooling2D(pool_size=(2, 2)),  # Capa de max pooling
        Dropout(0.25),  # Capa de dropout

        # Fully Connected Layers
        Flatten(),  # Capa para aplanar la salida de las capas convolucionales
        Dense(256, activation='relu'),  # Capa densa con 256 neuronas
        BatchNormalization(),  # Capa de normalización por lotes
        Dropout(0.5),  # Capa de dropout

        Dense(128, activation='relu'),  # Capa densa con 128 neuronas
        BatchNormalization(),  # Capa de normalización por lotes
        Dropout(0.5),  # Capa de dropout

        # Output Layer
        Dense(num_classes, activation='softmax')  # Capa de salida con activación softmax para clasificación multiclase
    ])
    return model  # Retorna el modelo creado

# Crear y compilar modelo
model = create_custom_cnn(input_shape, num_classes)  # Crea el modelo llamando a la función create_custom_cnn
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Compila el modelo con el optimizador Adam y tasa de aprendizaje 0.0005
              loss='categorical_crossentropy',  # Función de pérdida categórica cruzada
              metrics=['accuracy'])  # Métrica de precisión

# Generadores desde data_preprocessing (mismo preprocesado que Transfer)
train_generator = get_train_generator(batch_size=batch_size)
val_generator = get_validation_generator(batch_size=batch_size)

# Callbacks: guardar mejor modelo y parar si no hay mejora
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

# Entrenamiento (épocas máximas; puede parar antes por EarlyStopping)
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[checkpoint, early_stopping]
)