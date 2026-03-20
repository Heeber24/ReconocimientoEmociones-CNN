# -*- coding: utf-8 -*-
"""
División del dataset en train / validation / test.

Antes de copiar, elimina por completo la carpeta prepared_data para que no queden
archivos de corridas anteriores. Luego reparte desde DATA_DIR (FER_2013, AffectNet
o my_images según USE_KAGGLE_DATABASE y KAGGLE_DATASET).

Paso 2 del flujo (después de tener datos; ver README).
"""
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quiet_console

quiet_console.init()

from sklearn.model_selection import train_test_split
from project_config import DATASET_DIRS, DATA_SOURCE, PREPARED_DATA_DIR

if DATA_SOURCE not in DATASET_DIRS:
    print(f"Error: DATA_SOURCE inválido: {DATA_SOURCE!r}")
    print("Usa DATA_SOURCE = 'fer_2013', 'affectnet' o 'my_images' en scripts/project_config.py")
    sys.exit(1)

DATA_DIR = DATASET_DIRS[DATA_SOURCE]
OUTPUT_DIR = PREPARED_DATA_DIR

# --- CONFIGURACIÓN ---

# 2. Definimos las proporciones para la división del dataset.
# El 30% de los datos se reservará para validación y prueba.
VALIDATION_TEST_SPLIT_SIZE = 0.3
# De ese 30% restante, el 50% será para prueba (0.3 * 0.5 = 15% del total).
TEST_SPLIT_FROM_REMAINDER = 0.5
# El resultado final será: 70% train, 15% validation, 15% test.

# 3. Estado para controlar la aleatoriedad y asegurar que la división sea siempre la misma.
RANDOM_STATE = 42
# --------------------


def split_dataset():
    """
    Función principal que ejecuta todo el proceso de división del dataset.
    Siempre elimina la carpeta prepared_data anterior para que el split refleje solo
    las imágenes actuales en el origen (sin mezclar con corridas viejas).
    """
    # Verificación inicial: nos aseguramos de que el directorio de datos de origen exista.
    if not DATA_DIR.is_dir():
        print(f"Error: El directorio de datos de origen no existe: {DATA_DIR}")
        # sys.exit() detiene la ejecución del script si la carpeta principal no se encuentra.
        sys.exit()

    if OUTPUT_DIR.exists():
        print(f"Eliminando partición anterior: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Directorio de origen: {DATA_DIR}")
    print(f"Directorio de destino: {OUTPUT_DIR}")
    print("-" * 30)

    # Detección automática de las carpetas de emociones (clases).
    # .iterdir() lista todo el contenido y p.is_dir() filtra solo las carpetas.
    categories = [p for p in DATA_DIR.iterdir() if p.is_dir()]
    if not categories:
        print(f"Error: No se encontraron carpetas de emociones en {DATA_DIR}")
        sys.exit()
    
    print(f"Emociones detectadas: {[p.name for p in categories]}")
    print("-" * 30)

    # Procesamos cada categoría de emoción encontrada.
    for category_path in categories:
        category_name = category_path.name
        print(f"Procesando categoría: '{category_name}'...")

        # Se buscan imágenes con extensiones comunes. .glob() es una forma potente de buscar archivos.
        images = list(category_path.glob('*.[pP][nN][gG]')) + \
                 list(category_path.glob('*.[jJ][pP][gG]')) + \
                 list(category_path.glob('*.[jJ][pP][eE][gG]'))

        if not images:
            print(f"  -> No se encontraron imágenes en '{category_name}'. Saltando.")
            continue

        # --- División del Dataset ---
        # 1. Primera división: separamos el conjunto de entrenamiento del resto.
        train_files, val_test_files = train_test_split(
            images,
            test_size=VALIDATION_TEST_SPLIT_SIZE,
            random_state=RANDOM_STATE,
            stratify=[category_name] * len(images) # estratificación para mantener proporciones
        )

        # 2. Segunda división: separamos el resto en validación y prueba.
        val_files, test_files = train_test_split(
            val_test_files,
            test_size=TEST_SPLIT_FROM_REMAINDER,
            random_state=RANDOM_STATE,
            stratify=[category_name] * len(val_test_files) # estratificación
        )

        # Se agrupan los resultados para facilitar la copia de archivos.
        datasets = {
            'train': train_files,
            'validation': val_files,
            'test': test_files
        }

        # --- Copia de Archivos ---
        for set_name, file_list in datasets.items():
            # Se crea la ruta de destino final (ej: .../prepared_data/train/happy)
            destination_path = OUTPUT_DIR / set_name / category_name
            # Se crean las carpetas de destino de forma segura.
            destination_path.mkdir(parents=True, exist_ok=True)

            # Se copian los archivos de la fuente al destino.
            for file_path in file_list:
                # shutil.copy necesita rutas como strings, por eso usamos str().
                shutil.copy(str(file_path), str(destination_path))
        
        # Se imprime un resumen para esta categoría.
        print(f"  -> División completada: Entrenamiento({len(train_files)}), Validación({len(val_files)}), Prueba({len(test_files)})")

    print("-" * 30)
    print("✅ Proceso de división de datos completado exitosamente.")


# Este bloque asegura que la función solo se ejecute cuando el script es llamado directamente.
if __name__ == '__main__':
    split_dataset()