# -*- coding: utf-8 -*-
"""
Selector único de caminos de entrenamiento (1..6).

Uso:
    python scripts/run_model_path.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quiet_console

quiet_console.init()

from project_config import DATA_SOURCE, DATASET_DIRS
from training_utils import (
    MODEL_CAMINO_1,
    MODEL_CAMINO_2,
    MODEL_CAMINO_3,
    MODEL_CAMINO_4,
    MODEL_CAMINO_5,
    MODEL_CAMINO_6,
    print_training_prerequisites_banner,
    run_cnn_from_scratch_training,
    run_transfer_efficientnet_training,
    run_transfer_from_existing_model_training,
    tag_model_copy,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _is_kaggle_source(source: str) -> bool:
    return source in ("fer_2013", "affectnet")


def _run_step(script_name: str) -> None:
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)]
    print(f"\n>>> Ejecutando: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Falló {script_name} (exit_code={result.returncode})")


def _ask_path() -> int:
    print("\nSelecciona camino de entrenamiento:")
    print("  1) CNN desde cero con my_images")
    print("  2) EfficientNet transfer con my_images")
    print("  3) CNN desde cero con dataset base (fer_2013/affectnet)")
    print("  4) EfficientNet transfer con dataset base (fer_2013/affectnet)")
    print("  5) Transfer desde modelo camino 3 + my_images")
    print("  6) Transfer desde modelo camino 1 + dataset base")
    while True:
        raw = input("Camino (1..6): ").strip()
        if raw in {"1", "2", "3", "4", "5", "6"}:
            return int(raw)
        print("Valor inválido. Escribe 1, 2, 3, 4, 5 o 6.")


def _ask_model_base(expected_camino_suffix: str, label: str, default_path: Path) -> Path:
    print(f"\n{label}")
    print(f"Sugerido por regla del curso: modelo_camino_{expected_camino_suffix}.keras")
    print(f"Default: {default_path}")
    raw = input("Ruta del modelo base (.keras) [Enter = default]: ").strip()
    base = Path(raw) if raw else default_path
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    if not base.is_file():
        raise FileNotFoundError(f"No existe el modelo base: {base}")

    expected_token = f"camino_{expected_camino_suffix}"
    if expected_token not in base.name.lower():
        print("\n[Advertencia] Ese modelo NO coincide con la regla del profesor.")
        print(f"  - Para este camino se recomienda usar: *{expected_token}*.keras")
        print("  - Se puede continuar, pero rompe la regla del programa.")
        ans = input("¿Deseas continuar de todos modos? (s/N): ").strip().lower()
        if ans not in {"s", "si", "sí", "y", "yes"}:
            raise RuntimeError("Operación cancelada por el usuario.")
    return base


def _run_training_by_path(path_num: int) -> Path:
    if path_num == 1:
        print_training_prerequisites_banner(transfer_efficientnet=False)
        out, acc = run_cnn_from_scratch_training(MODEL_CAMINO_1)
        tag_model_copy(out, "modelo_camino_1", acc)
        return out
    if path_num == 2:
        print_training_prerequisites_banner(transfer_efficientnet=True)
        out, acc = run_transfer_efficientnet_training(MODEL_CAMINO_2)
        tag_model_copy(out, "modelo_camino_2", acc)
        return out
    if path_num == 3:
        print_training_prerequisites_banner(transfer_efficientnet=False)
        out, acc = run_cnn_from_scratch_training(MODEL_CAMINO_3)
        tag_model_copy(out, "modelo_camino_3", acc)
        return out
    if path_num == 4:
        print_training_prerequisites_banner(transfer_efficientnet=True)
        out, acc = run_transfer_efficientnet_training(MODEL_CAMINO_4)
        tag_model_copy(out, "modelo_camino_4", acc)
        return out
    if path_num == 5:
        print_training_prerequisites_banner(transfer_efficientnet=False)
        base = _ask_model_base("3", "Camino 5: base recomendada = modelo camino 3", MODEL_CAMINO_3)
        out, acc = run_transfer_from_existing_model_training(base, MODEL_CAMINO_5)
        tag_model_copy(out, "modelo_camino_5", acc)
        return out

    print_training_prerequisites_banner(transfer_efficientnet=False)
    base = _ask_model_base("1", "Camino 6: base recomendada = modelo camino 1", MODEL_CAMINO_1)
    out, acc = run_transfer_from_existing_model_training(base, MODEL_CAMINO_6)
    tag_model_copy(out, "modelo_camino_6", acc)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Selector único de caminos (1..6).")
    parser.add_argument("--path", type=int, default=None, help="Camino a ejecutar (1..6).")
    parser.add_argument("--skip-realtime", action="store_true", help="No preguntar ni abrir realtime.")
    args = parser.parse_args()

    print("\n=== run_model_path.py ===")
    print(f"DATA_SOURCE actual: {DATA_SOURCE}")
    if DATA_SOURCE not in DATASET_DIRS:
        raise RuntimeError("DATA_SOURCE inválido en project_config.py")

    if args.path is not None:
        if args.path not in {1, 2, 3, 4, 5, 6}:
            raise RuntimeError("--path debe ser un valor entre 1 y 6.")
        path_num = args.path
    else:
        path_num = _ask_path()
    needs_kaggle = path_num in {3, 4, 6}

    if needs_kaggle and not _is_kaggle_source(DATA_SOURCE):
        raise RuntimeError(
            f"El camino {path_num} requiere DATA_SOURCE='fer_2013' o 'affectnet'. "
            f"Actual: {DATA_SOURCE!r}."
        )
    if (not needs_kaggle) and DATA_SOURCE != "my_images":
        print(
            f"[Aviso] El camino {path_num} suele usarse con my_images. DATA_SOURCE actual: {DATA_SOURCE}."
        )

    print("\nNota:")
    print("  Este script solo ejecuta ENTRENAMIENTO del camino elegido.")
    print("  Debes correr antes:")
    print("    python scripts/data_split.py")
    print("    python scripts/data_preprocessing.py")
    trained_path = _run_training_by_path(path_num)

    print(f"\n✅ Listo. Camino {path_num} terminado.")
    print(f"Modelo principal: {trained_path}")
    if not args.skip_realtime:
        run_rt = input("¿Abrir realtime ahora? (s/N): ").strip().lower()
        if run_rt in {"s", "si", "sí", "y", "yes"}:
            subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "realtime_emotion_recognition.py")],
                cwd=str(PROJECT_ROOT),
            )


if __name__ == "__main__":
    main()

