# Importar modelos preentrenados de Keras y librerías necesarias
import tensorflow as tf  # Importa la biblioteca TensorFlow para aprendizaje profundo
from tensorflow.keras.applications import EfficientNetB0   # type: ignore # Importa el modelo VGG16 preentrenado desde Keras Applications
#from tensorflow.keras.applications import VGG16  # type: ignore # Importa el modelo VGG16 preentrenado desde Keras Applications
from tensorflow.keras.models import Sequential  # type: ignore # Importa la clase Sequential para construir modelos secuenciales
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization  # type: ignore # Importa capas necesarias para la construcción del modelo
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore # Importa ImageDataGenerator para la generación de datos con aumentación
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # type: ignore # Importa callbacks para optimizar el entrenamiento
import os  # Importa la biblioteca os para interactuar con el sistema operativo
import numpy as np  # Importa NumPy para operaciones numéricas

# Directorios de datos
train_dir = r'C:...\ReconocimientoEmociones(CNN)\data\Tus_Imagenes\prepared_data\train'  # Ruta al directorio de entrenamiento
validation_dir = r'C:...\ReconocimientoEmociones(CNN)\data\Tus_Imagenes\prepared_data\validation'  # Ruta al directorio de validación
test_dir = r'C:...\ReconocimientoEmociones(CNN)\data\Tus_Imagenes\prepared_data\test'  # Ruta al directorio de prueba

# Verificar directorios
for path in [train_dir, validation_dir, test_dir]:  # Itera sobre los directorios
    if not os.path.exists(path):  # Verifica si el directorio existe
        print(f"Error: El directorio no existe: {path}")  # Imprime un mensaje de error si no existe
        exit()  # Sale del programa

# Generadores de datos con aumentación
train_datagen = ImageDataGenerator(  # Crea un generador de datos para entrenamiento con aumentación
    rescale=1. / 255,  # Normaliza los píxeles al rango [0, 1]
    rotation_range=30,  # Aplica rotaciones aleatorias de hasta 30 grados
    width_shift_range=0.3,  # Aplica desplazamientos horizontales de hasta 30%
    height_shift_range=0.3,  # Aplica desplazamientos verticales de hasta 30%
    horizontal_flip=True,  # Aplica volteos horizontales aleatorios
    vertical_flip=True,  # Aplica volteos verticales aleatorios
    brightness_range=[0.7, 1.3],  # Ajusta el brillo aleatoriamente entre 70% y 130%
    zoom_range=0.3,  # Aplica zoom aleatorio de hasta 30%
    shear_range=0.3,  # Aplica deformaciones aleatorias de hasta 30%
    fill_mode='nearest'  # Rellena los píxeles vacíos con el píxel más cercano
)

val_datagen = ImageDataGenerator(rescale=1. / 255)  # Crea un generador de datos para validación con solo normalización

train_generator = train_datagen.flow_from_directory(  # Crea un generador de datos de entrenamiento a partir del directorio
    train_dir,  # Directorio de entrenamiento
    target_size=(224, 224),  # Redimensiona las imágenes a 224x224 píxeles
    batch_size=32,  # Tamaño del lote
    class_mode='categorical',  # Modo de clasificación multicategoría
    color_mode='rgb',  # Imágenes en color RGB
    seed=42  # Semilla para reproducibilidad
)

validation_generator = val_datagen.flow_from_directory(  # Crea un generador de datos de validación a partir del directorio
    validation_dir,  # Directorio de validación
    target_size=(224, 224),  # Redimensiona las imágenes a 224x224 píxeles
    batch_size=32,  # Tamaño del lote
    class_mode='categorical',  # Modo de clasificación multicategoría
    color_mode='rgb',  # Imágenes en color RGB
    seed=42  # Semilla para reproducibilidad
)

# Modelo VGG16 preentrenado
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # Carga el modelo VGG16 preentrenado con pesos de ImageNet

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

# Generador de datos de prueba
test_datagen = ImageDataGenerator(rescale=1. / 255)  # Crea un generador de datos de prueba con solo normalización

test_generator = test_datagen.flow_from_directory(  # Crea un generador de datos de prueba a partir del directorio
    test_dir,  # Directorio de prueba
    target_size=(224, 224),  # Redimensiona las imágenes a 224x224 píxeles
    batch_size=32,  # Tamaño del lote
    class_mode='categorical',  # Modo de clasificación multicategoría
    shuffle=False,  # No mezcla los datos de prueba
    color_mode='rgb',  # Imágenes en color RGB
    seed=42  # Semilla para reproducibilidad
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Callback para detener el entrenamiento temprano
checkpoint = ModelCheckpoint(  # Callback para guardar el mejor modelo

    r'C:...\ReconocimientoEmociones(CNN)\models\emotion_recognition_EfficientNetB0_model.keras',  # Ruta para guardar el modelosave_best_only=True,  # Guarda solo el mejor modelo
    monitor='val_loss',  # Monitorea la pérdida de validación
    mode='min'  # Modo mínimo para la pérdida de validación
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