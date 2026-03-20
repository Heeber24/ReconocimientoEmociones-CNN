# -*- coding: utf-8 -*-
"""
Preprocesado de datos para entrenamiento: normalización y data augmentation.

Este script es el ÚNICO lugar donde se define el preprocesado (rescale 1/255,
data augmentation: rotación, shift, flip, brillo, zoom, etc.). Tanto
los caminos con transfer learning como los caminos de red desde cero
importan los generadores desde aquí y solo se encargan de construir y
entrenar el modelo; no duplican lógica de preprocesado.

Al ejecutarlo como script: valida train/ y validation/, muestra conteo por
emoción; opción --data-root para validar otra ruta (repo externo). Paso 3 del flujo.
Flujo completo: ver README en la raíz del proyecto.
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

# =============================================================================
# CONFIGURA AQUÍ
#
# Para este proyecto:
# - Si usas datos "propias" (OpenCV webcam) -> USE_KAGGLE_DATABASE=False -> BGR en disco
# - Si usas datos Kaggle (FER_2013/AffectNet) -> USE_KAGGLE_DATABASE=True -> ya vienen en RGB
#
# Nota: el nombre del dataset Kaggle (FER_2013 vs AffectNet) no cambia el preprocesado
# aquí, porque en ambos casos esperamos imágenes listas para RGB.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# True = preprocesar datasets Kaggle (FER_2013 o AffectNet)
# False = preprocesar datos propios capturados (data/my_images)
USE_KAGGLE_DATABASE = True

# Solo informativo (si quieres entrenar con AffectNet, pon USE_KAGGLE_DATABASE=True
# igual; el origen ya lo eliges en data_split.py con KAGGLE_DATASET).
KAGGLE_DATASET = "fer_2013"  # "fer_2013" o "affectnet"

PREPARED_DATA = PROJECT_ROOT / "data" / "prepared_data"
DATA_ROOT = PREPARED_DATA
TRAIN_DIR = DATA_ROOT / "train"
VALIDATION_DIR = DATA_ROOT / "validation"
TEST_DIR = DATA_ROOT / "test"
IMAGES_ARE_BGR = not USE_KAGGLE_DATABASE  # BGR en propias; Kaggle (FER_2013/AffectNet) ya en RGB
# =============================================================================

# Parámetros de imagen usados en todo el proyecto (captura, transfer ImageNet, CNN desde cero)
TARGET_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32
SEED = 42

# Las imágenes guardadas con OpenCV (data_collection) están en BGR; convertimos a RGB para el modelo.
def bgr_to_rgb(img):
    return img[..., ::-1].copy()


def _preprocess_fn():
    return bgr_to_rgb if IMAGES_ARE_BGR else None


def get_train_datagen():
    """Generador para entrenamiento: normalización + data augmentation."""
    return ImageDataGenerator(
        rescale=1.0 / 255,
        preprocessing_function=_preprocess_fn(),
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        # vertical_flip=False: no usar en rostros (las caras no son simétricas arriba/abajo)
        brightness_range=[0.7, 1.3],
        zoom_range=0.3,
        shear_range=0.3,
        fill_mode="nearest",
    )


def get_val_datagen():
    """Generador para validación: solo normalización."""
    return ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=_preprocess_fn())


def get_test_datagen():
    """Generador para prueba: solo normalización."""
    return ImageDataGenerator(rescale=1.0 / 255, preprocessing_function=_preprocess_fn())


def get_train_generator(
    data_root=None,
    target_size=TARGET_SIZE,
    batch_size=DEFAULT_BATCH_SIZE,
    seed=SEED,
):
    """Generador de lotes de entrenamiento. data_root=None usa DATA_ROOT de este script."""
    root = Path(data_root) if data_root is not None else Path(DATA_ROOT)
    train_dir = root / "train"
    datagen = get_train_datagen()
    return datagen.flow_from_directory(
        str(train_dir),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="rgb",
        seed=seed,
    )


def get_validation_generator(
    data_root=None,
    target_size=TARGET_SIZE,
    batch_size=DEFAULT_BATCH_SIZE,
    seed=SEED,
):
    """Generador de lotes de validación. data_root=None usa DATA_ROOT de este script."""
    root = Path(data_root) if data_root is not None else Path(DATA_ROOT)
    val_dir = root / "validation"
    datagen = get_val_datagen()
    return datagen.flow_from_directory(
        str(val_dir),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="rgb",
        seed=seed,
    )


def get_test_generator(
    data_root=None,
    target_size=TARGET_SIZE,
    batch_size=DEFAULT_BATCH_SIZE,
    shuffle=False,
    seed=SEED,
):
    """Generador de lotes de prueba. data_root=None usa DATA_ROOT de este script."""
    root = Path(data_root) if data_root is not None else Path(DATA_ROOT)
    test_dir = root / "test"
    datagen = get_test_datagen()
    return datagen.flow_from_directory(
        str(test_dir),
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=shuffle,
        seed=seed,
    )


def validate_and_report(data_root=None):
    """Valida que existan train/ y validation/ y muestra conteo por emoción."""
    root = Path(data_root) if data_root is not None else Path(DATA_ROOT)
    train_dir = root / "train"
    val_dir = root / "validation"
    test_dir = root / "test"

    errors = []
    if not train_dir.is_dir():
        errors.append(f"No existe directorio de entrenamiento: {train_dir}")
    if not val_dir.is_dir():
        errors.append(f"No existe directorio de validación: {val_dir}")

    if errors:
        for e in errors:
            print(e)
        return False

    def count_per_class(directory):
        if not directory.is_dir():
            return {}
        classes = sorted([p.name for p in directory.iterdir() if p.is_dir()])
        counts = {}
        for c in classes:
            path = directory / c
            n = len(list(path.glob("*.png")) + list(path.glob("*.jpg")) + list(path.glob("*.jpeg")))
            counts[c] = n
        return counts

    print("Preprocesado de datos – validación")
    print("-" * 50)
    print(f"Raíz de datos: {root}")
    print()
    print("Train:")
    for cls, n in count_per_class(train_dir).items():
        print(f"  {cls}: {n} imágenes")
    print(f"  Total train: {sum(count_per_class(train_dir).values())}")
    print()
    print("Validation:")
    for cls, n in count_per_class(val_dir).items():
        print(f"  {cls}: {n} imágenes")
    print(f"  Total validation: {sum(count_per_class(val_dir).values())}")
    if test_dir.is_dir():
        print()
        print("Test:")
        for cls, n in count_per_class(test_dir).items():
            print(f"  {cls}: {n} imágenes")
        print(f"  Total test: {sum(count_per_class(test_dir).values())}")
    print()

    # Crear generadores y comprobar que cargan
    try:
        tg = get_train_generator(data_root=str(root))
        vg = get_validation_generator(data_root=str(root))
        print(f"Generadores creados: train {tg.samples} muestras, validation {vg.samples} muestras.")
        print("Preprocesado listo. Puedes entrenar con scripts de paths (generate_model_path_1..6).")
    except Exception as e:
        print(f"Error al crear generadores: {e}")
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validar datos y comprobar preprocesado (generadores). Por defecto usa data/prepared_data."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Ruta a la raíz de datos (carpeta que contiene train/, validation/, test/). "
        "Si no se indica, se usa DATA_ROOT definido arriba en este script.",
    )
    args = parser.parse_args()
    success = validate_and_report(data_root=args.data_root)
    sys.exit(0 if success else 1)
