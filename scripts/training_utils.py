# -*- coding: utf-8 -*-
"""
Entrenamiento compartido: CNN desde cero y transfer con EfficientNetB0 (ImageNet).
El proyecto está pensado solo para EfficientNet; otro backbone implica tocar código.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0  # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Lambda,
    MaxPooling2D,
    Rescaling,
)
from tensorflow.keras.models import Sequential, load_model  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))

# =============================================================================
# MODELOS POR CAMINO (un archivo principal por número; no se pisan entre sí)
#
# Carpeta: models/  |  Cada generate_model_path_N guarda en modelo_camino_N.keras
#
# Camino 5: transfer con datos propios → base típica = MODEL_CAMINO_3 (CNN FER).
# Camino 6: transfer con FER → base típica = MODEL_CAMINO_1 (CNN propias).
#
# Las rutas de datos (train/val/test) vienen de data_preprocessing (split/preprocess).
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_CAMINO_1 = MODELS_DIR / "modelo_camino_1.keras"
MODEL_CAMINO_2 = MODELS_DIR / "modelo_camino_2.keras"
MODEL_CAMINO_3 = MODELS_DIR / "modelo_camino_3.keras"
MODEL_CAMINO_4 = MODELS_DIR / "modelo_camino_4.keras"
MODEL_CAMINO_5 = MODELS_DIR / "modelo_camino_5.keras"
MODEL_CAMINO_6 = MODELS_DIR / "modelo_camino_6.keras"
# =============================================================================

from data_preprocessing import (  # noqa: E402
    TEST_DIR,
    TRAIN_DIR,
    VALIDATION_DIR,
    get_test_generator,
    get_train_generator,
    get_validation_generator,
)


def efficientnet_preprocess_fn(x):
    """Para capa Lambda serializable (Keras 3) y custom_objects al cargar."""
    return efficientnet_preprocess_input(x)


def print_training_prerequisites_banner(*, transfer_efficientnet: bool = False) -> None:
    print("\n" + "=" * 70)
    print("Este script SOLO entrena. No ejecuta data_split ni data_preprocessing.")
    print("Orden que debes haber corrido tú antes:")
    print("  1. python scripts/data_split.py   (borra y regenera prepared_data)")
    print("  2. python scripts/data_preprocessing.py")
    if transfer_efficientnet:
        print("")
        print("Transfer: dentro del modelo se aplica preprocess_input de EfficientNet;")
        print("data_preprocessing.py sigue siendo necesario para los generadores desde disco.")
    print("=" * 70 + "\n")


def ensure_kaggle_flag(expected: bool, label: str, use_kaggle_database: bool) -> None:
    """Comprueba que la bandera del script de camino coincide con la fase (Kaggle vs propias)."""
    if use_kaggle_database != expected:
        print(f"[{label}] Inconsistencia de USE_KAGGLE_DATABASE.")
        print(f"  - Este paso espera: USE_KAGGLE_DATABASE = {expected}")
        print(f"  - Tienes en este script: {use_kaggle_database}")
        print("  Alinea data_split.py, data_preprocessing.py y el generate_model_path_*.py.")
        sys.exit(1)


def _check_train_dirs() -> None:
    for path in [TRAIN_DIR, VALIDATION_DIR, TEST_DIR]:
        if not path.exists():
            print(f"Error: El directorio no existe: {path}")
            print("Orden esperado antes de entrenar:")
            print("  1. python scripts/data_split.py")
            print("  2. python scripts/data_preprocessing.py")
            sys.exit(1)


def run_transfer_efficientnet_training(save_path: Path) -> tuple[Path, float]:
    """
    Transfer learning con EfficientNetB0 (ImageNet).
    save_path: donde guardar el mejor modelo (p. ej. MODEL_CAMINO_2 o MODEL_CAMINO_4).
    Devuelve (ruta del checkpoint, accuracy en test tras fine-tuning).
    """
    _check_train_dirs()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    input_shape = (224, 224, 3)
    model_transfer_path = save_path

    batch_size = 32
    train_generator = get_train_generator(target_size=input_shape[:2], batch_size=batch_size, seed=42)
    validation_generator = get_validation_generator(target_size=input_shape[:2], batch_size=batch_size, seed=42)
    test_generator = get_test_generator(target_size=input_shape[:2], batch_size=batch_size, shuffle=False, seed=42)
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size
    print("Usando base: EfficientNetB0")
    print(f"Steps por época: {steps_per_epoch}, validation steps: {validation_steps}")

    num_classes = train_generator.num_classes
    class_indices = train_generator.class_indices
    counts = np.zeros(num_classes)
    for class_name, idx in class_indices.items():
        folder = Path(TRAIN_DIR) / class_name
        if folder.is_dir():
            counts[idx] = len(list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png")))
    if counts.min() > 0:
        class_weights = {i: counts.sum() / (num_classes * counts[i]) for i in range(num_classes)}
        print("Pesos por clase (desbalance):", class_weights)
    else:
        class_weights = None

    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = Sequential([
        Rescaling(255.0, name="to_0_255"),
        Lambda(efficientnet_preprocess_fn, name="backbone_preprocess"),
        base_model,
        GlobalAveragePooling2D(),
        Dense(1024, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(str(model_transfer_path), save_best_only=True, monitor="val_loss", mode="min")
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=5, min_lr=1e-6)

    np.random.seed(42)
    tf.random.set_seed(42)

    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        class_weight=class_weights,
        callbacks=[early_stopping, checkpoint, lr_scheduler],
    )

    scores = model.evaluate(test_generator, verbose=0)
    print(f"Precisión del modelo en el conjunto de prueba: {scores[1] * 100:.2f}%")

    base_model.trainable = True
    fine_tune_at = int(len(base_model.layers) * 0.8)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        class_weight=class_weights,
        callbacks=[early_stopping, checkpoint, lr_scheduler],
    )
    scores = model.evaluate(test_generator, verbose=0)
    acc = float(scores[1])
    print(f"Precisión después del ajuste fino: {acc * 100:.2f}%")
    return model_transfer_path, acc


def load_model_for_continue_training(base_model_path: Path):
    """Carga CNN propia o modelo EfficientNet previo."""
    try:
        return load_model(str(base_model_path), safe_mode=False)
    except Exception:
        return load_model(
            str(base_model_path),
            safe_mode=False,
            custom_objects={"preprocess_fn": efficientnet_preprocess_fn},
        )


def run_transfer_from_existing_model_training(base_model_path: Path, save_path: Path) -> tuple[Path, float]:
    """
    Quita la cabeza del modelo base, congela el resto y entrena nueva softmax.
    save_path: archivo de salida (p. ej. MODEL_CAMINO_5 o MODEL_CAMINO_6).
    """
    _check_train_dirs()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not base_model_path.exists():
        print(f"No existe modelo base: {base_model_path}")
        sys.exit(1)
    print("Cargando modelo como base:", base_model_path)
    model = load_model_for_continue_training(base_model_path)
    model.pop()
    for layer in model.layers:
        layer.trainable = False

    train_gen = get_train_generator(target_size=(224, 224), batch_size=32, seed=42)
    val_gen = get_validation_generator(target_size=(224, 224), batch_size=32, seed=42)
    test_gen = get_test_generator(target_size=(224, 224), batch_size=32, shuffle=False, seed=42)

    model.add(Dense(train_gen.num_classes, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    checkpoint = ModelCheckpoint(str(save_path), save_best_only=True, monitor="val_loss", mode="min")
    early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    lr_reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=5, min_lr=1e-6)

    np.random.seed(42)
    tf.random.set_seed(42)
    model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        epochs=50,
        validation_data=val_gen,
        validation_steps=val_gen.samples // val_gen.batch_size,
        callbacks=[checkpoint, early, lr_reduce],
    )
    scores = model.evaluate(test_gen, verbose=0)
    acc = float(scores[1])
    print(f"Precisión en test: {acc * 100:.2f}%")
    print(f"Modelo guardado en: {save_path}")
    return save_path, acc


def run_cnn_from_scratch_training(save_path: Path) -> tuple[Path, float]:
    """CNN desde cero. save_path: p. ej. MODEL_CAMINO_1 o MODEL_CAMINO_3."""
    _check_train_dirs()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("Entrenando CNN desde cero...")
    input_shape = (224, 224, 3)
    batch_size = 64
    epochs = 30
    early_stopping_patience = 10

    train_generator = get_train_generator(batch_size=batch_size)
    val_generator = get_validation_generator(batch_size=batch_size)
    test_generator = get_test_generator(batch_size=batch_size, shuffle=False, seed=42)

    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint = ModelCheckpoint(str(save_path), monitor="val_accuracy", save_best_only=True, mode="max")
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        restore_best_weights=True,
    )
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping],
    )
    scores = model.evaluate(test_generator, verbose=0)
    acc = float(scores[1])
    print(f"Precisión en test: {acc * 100:.2f}%")
    print(f"Modelo guardado en: {save_path}")
    return save_path, acc


def tag_model_copy(source: Path, name_stem: str, accuracy: float) -> Path:
    """Copia con fecha y accuracy en el nombre del archivo."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    stem = name_stem[:-6] if name_stem.lower().endswith(".keras") else name_stem
    stem = stem.strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    acc_part = f"{accuracy * 100:.2f}".replace(".", "p")
    target = MODELS_DIR / f"{stem}_{timestamp}_acc{acc_part}pct.keras"
    shutil.copy2(source, target)
    print(f"Copia etiquetada: {target}")
    return target

