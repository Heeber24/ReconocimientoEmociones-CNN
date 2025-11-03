# Importamos las bibliotecas necesarias
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Generador de datos con aumentación # type: ignore
import os  # Manejo de directorios

# Rutas a los directorios de entrenamiento y validación
train_dir = r'C:...\ReconocimientoEmociones(CNN)\data\Tus_Imagenes\prepared_data\train'
validation_dir = r'C:...\ReconocimientoEmociones(CNN)\data\Tus_Imagenes\prepared_data\validation'


# Verificar existencia de directorios
if not os.path.exists(train_dir):
    print(f"Error: El directorio de entrenamiento no existe: {train_dir}")
if not os.path.exists(validation_dir):
    print(f"Error: El directorio de validación no existe: {validation_dir}")

# Generador de datos para entrenamiento con aumentación
train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalizar píxeles al rango [0, 1]   
    rotation_range=20,  # Rotación aleatoria hasta 20 grados
    width_shift_range=0.1,  # Desplazamiento horizontal limitado (10%)
    height_shift_range=0.1,  # Desplazamiento vertical limitado (10%)
    horizontal_flip=True,  # Volteo horizontal aleatorio
    brightness_range=[0.8, 1.2],  # Variación de brillo entre 80% y 120%
)

# Generador de datos para validación (sin aumentación adicional)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# Cargar imágenes de entrenamiento con aumentación
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Tamaño requerido 
    batch_size=32,
    class_mode='categorical',  # Clasificación multicategoría
    color_mode='rgb'  # Imágenes en color
)

# Cargar imágenes de validación
validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),  # Tamaño consistente con entrenamiento
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb'
)

# Imprimir estadísticas de los datos cargados
print(f"Imágenes de entrenamiento encontradas: {train_generator.samples}")
print(f"Imágenes de validación encontradas: {validation_generator.samples}")



print("\nGeneradores de datos listos para entrenar el modelo.")
