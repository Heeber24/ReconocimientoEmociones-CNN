import tensorflow as tf  # Importa la biblioteca TensorFlow para aprendizaje profundo
from tensorflow.keras.models import Sequential  # Importa la clase Sequential para crear modelos secuenciales  # type: ignore
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense  # Importa capas necesarias para la CNN  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Importa ImageDataGenerator para aumento de datos  # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint  # Importa ModelCheckpoint para guardar el mejor modelo  # type: ignore

# Configuración
input_shape = (224, 224, 3)  # Define la forma de entrada de las imágenes (alto, ancho, canales)
num_classes = 4  # Define el número de clases de salida
batch_size = 64  # Define el tamaño del lote para el entrenamiento

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

# Aumento de datos
datagen = ImageDataGenerator(  # Crea un generador de datos con aumentación
    rescale=1./255,  # Normaliza los píxeles al rango [0, 1]
    rotation_range=15,  # Aplica rotaciones aleatorias de hasta 15 grados
    width_shift_range=0.1,  # Aplica desplazamientos horizontales aleatorios
    height_shift_range=0.1,  # Aplica desplazamientos verticales aleatorios
    horizontal_flip=True  # Aplica volteos horizontales aleatorios
)

# Cargar dataset
train_generator = datagen.flow_from_directory(  # Crea un generador de datos de entrenamiento a partir del directorio
    r'C:...\ReconocimientoEmociones(CNN)\data\Tus_imagenes\prepared_data\train',  # Ruta al directorio de entrenamiento
    target_size=(224, 224),  # Redimensiona las imágenes a 224x224
    batch_size=batch_size,  # Tamaño del lote
    class_mode='categorical'  # Modo de clasificación multiclase
)

val_generator = datagen.flow_from_directory(  # Crea un generador de datos de validación a partir del directorio
    r'C:...\ReconocimientoEmociones(CNN)\data\Tus_imagenes\prepared_data\validation',  # Ruta al directorio de validación
    target_size=(224, 224),  # Redimensiona las imágenes a 224x224
    batch_size=batch_size,  # Tamaño del lote
    class_mode='categorical'  # Modo de clasificación multiclase
)

# Guardar solo el mejor modelo
checkpoint = ModelCheckpoint(  # Crea un callback para guardar el mejor modelo
    r'C:...\ReconocimientoEmociones(CNN)\models\emotion_recognition_Personal.keras',  # Ruta para guardar el modelo
    monitor='val_accuracy',  # Monitorea la precisión de validación
    save_best_only=True,  # Guarda solo el mejor modelo
    mode='max'  # Guarda el modelo cuando la precisión de validación es máxima
)

# Entrenamiento
model.fit(train_generator,  # Entrena el modelo con el generador de datos de entrenamiento
          validation_data=val_generator,  # Usa el generador de datos de validación para la validación
          epochs=20,  # Número de épocas de entrenamiento
          callbacks=[checkpoint])  # Usa el callback de checkpoint para guardar el mejor modelo