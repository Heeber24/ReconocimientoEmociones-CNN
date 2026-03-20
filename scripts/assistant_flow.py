# -*- coding: utf-8 -*-
"""
Flujo guiado completo para alumnos:
1) Elegir camino
2) Elegir/validar fuente de datos
3) Opcional captura de my_images
4) Ejecutar split + preprocessing + entrenamiento
5) Opcional realtime

Uso:
    python scripts/assistant_flow.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PROJECT_CONFIG = SCRIPTS_DIR / "project_config.py"

sys.path.insert(0, str(SCRIPTS_DIR))
import quiet_console

quiet_console.init()

# Mismas clases que data_collection / split esperan bajo data/my_images/<emoción>/
MY_IMAGE_EMOTIONS = ["angry", "happy", "neutral", "surprise"]
# Mínimo sugerido por clase para no ir al split “en vacío”
MIN_IMAGES_PER_CLASS = 5


def _run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode
    if rc != 0:
        raise RuntimeError(f"Comando falló (exit_code={rc}): {' '.join(cmd)}")


def _run_data_collection() -> int:
    """Ejecuta captura; devuelve código de salida (0 = proceso terminó sin crash)."""
    cmd = [sys.executable, str(SCRIPTS_DIR / "data_collection.py")]
    print(f"\n>>> {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def _ask_path() -> int:
    print("\nSeleccione camino (1..6):")
    for i, text in [
        (1, "CNN desde cero con my_images"),
        (2, "EfficientNet transfer con my_images"),
        (3, "CNN desde cero con dataset base"),
        (4, "EfficientNet transfer con dataset base"),
        (5, "Transfer desde camino 3 + my_images"),
        (6, "Transfer desde camino 1 + dataset base"),
    ]:
        print(f"  {i}) {text}")
    while True:
        raw = input("Camino: ").strip()
        if raw in {"1", "2", "3", "4", "5", "6"}:
            return int(raw)
        print("Valor inválido.")


def _ask_source(path_num: int) -> str:
    if path_num in {1, 2, 5}:
        return "my_images"

    print("\nEste camino requiere dataset base.")
    print("  1) fer_2013")
    print("  2) affectnet")
    while True:
        raw = input("Fuente (1/2): ").strip()
        if raw == "1":
            return "fer_2013"
        if raw == "2":
            return "affectnet"
        print("Valor inválido.")


def _write_data_source(new_source: str) -> None:
    text = PROJECT_CONFIG.read_text(encoding="utf-8")
    lines = text.splitlines()
    out: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith("DATA_SOURCE = "):
            out.append(f'DATA_SOURCE = "{new_source}"')
            replaced = True
        else:
            out.append(line)
    if not replaced:
        raise RuntimeError("No se encontró DATA_SOURCE en project_config.py")
    PROJECT_CONFIG.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"DATA_SOURCE actualizado a: {new_source}")


def _ensure_source_exists(source: str) -> None:
    target = {
        "fer_2013": PROJECT_ROOT / "data" / "FER_2013",
        "affectnet": PROJECT_ROOT / "data" / "AffectNet",
        "my_images": PROJECT_ROOT / "data" / "my_images",
    }[source]
    if not target.exists():
        raise FileNotFoundError(f"No existe la carpeta de datos: {target}")
    if not any(target.iterdir()):
        raise RuntimeError(f"La carpeta existe pero está vacía: {target}")


def _count_images_in_folder(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def _my_images_counts(base: Path) -> dict[str, int]:
    return {e: _count_images_in_folder(base / e) for e in MY_IMAGE_EMOTIONS}


def _my_images_total(counts: dict[str, int]) -> int:
    return sum(counts.values())


def _my_images_sufficient(counts: dict[str, int]) -> bool:
    return all(c >= MIN_IMAGES_PER_CLASS for c in counts.values())


def _print_my_images_status(counts: dict[str, int]) -> None:
    print("\n--- Estado actual de data/my_images ---")
    for k, v in counts.items():
        mark = "✓" if v >= MIN_IMAGES_PER_CLASS else "!"
        print(f"  {mark} {k}: {v} imágenes")
    print(f"  Total: {_my_images_total(counts)}")


def _after_capture_decision(base: Path) -> Literal["again", "split", "restart", "exit"]:
    """
    Tras cerrar data_collection: no se asume que quieras seguir al split.
    """
    while True:
        counts = _my_images_counts(base)
        _print_my_images_status(counts)
        total = _my_images_total(counts)
        ok = _my_images_sufficient(counts)

        if not ok:
            low = [k for k, v in counts.items() if v < MIN_IMAGES_PER_CLASS]
            print(
                f"\n⚠ Para un entrenamiento razonable se sugieren al menos "
                f"{MIN_IMAGES_PER_CLASS} imágenes por emoción. Revisa: {', '.join(low)}"
            )
        print(
            "\n¿Qué quieres hacer ahora?\n"
            "  1) Capturar de nuevo (abrir otra vez la webcam)\n"
            "  2) Continuar y ejecutar split + preprocesado + entrenamiento con lo que hay\n"
            "  3) Reiniciar el asistente desde el inicio (volver a elegir camino)\n"
            "  4) Salir sin ejecutar split"
        )
        raw = input("Opción [1-4]: ").strip()
        if raw == "1":
            return "again"
        if raw == "2":
            if total == 0:
                print("No hay imágenes en my_images. Captura primero (opción 1).")
                continue
            if not ok:
                sure = input(
                    "Hay pocas imágenes en alguna clase. ¿Continuar igual con el split? (s/N): "
                ).strip().lower()
                if sure not in {"s", "si", "sí", "y", "yes"}:
                    continue
            return "split"
        if raw == "3":
            return "restart"
        if raw == "4":
            return "exit"
        print("Opción inválida.")


def _handle_my_images_flow() -> Literal["continue", "restart"]:
    """
    Gestiona captura y evita seguir al split al cerrar la ventana sin confirmar.
    """
    base = PROJECT_ROOT / "data" / "my_images"
    base.mkdir(parents=True, exist_ok=True)

    while True:
        counts = _my_images_counts(base)
        total = _my_images_total(counts)

        if total == 0:
            print("\nmy_images no tiene imágenes en las carpetas de emociones (vacío).")
            a = input("¿Capturar ahora con la webcam? (S/n): ").strip().lower()
            if a in {"n", "no"}:
                sub = input("  (1) Reiniciar asistente desde el inicio  (2) Salir: ").strip()
                if sub == "1":
                    return "restart"
                sys.exit(0)
            rc = _run_data_collection()
            if rc != 0:
                print(f"\n[Nota] data_collection terminó con código {rc}.")
            while True:
                decision = _after_capture_decision(base)
                if decision == "again":
                    rc = _run_data_collection()
                    if rc != 0:
                        print(f"\n[Nota] data_collection terminó con código {rc}.")
                    continue
                if decision == "split":
                    return "continue"
                if decision == "restart":
                    return "restart"
                sys.exit(0)

        # Ya hay al menos una imagen en alguna carpeta
        print("\nmy_images ya tiene datos guardados.")
        _print_my_images_status(counts)
        recap = input(
            "\n¿Deseas abrir de nuevo la captura (webcam)? (s/N)\n"
            "  (N = usar solo lo que ya está en disco, sin abrir la cámara)\n"
            "Tu elección: "
        ).strip().lower()

        if recap in {"s", "si", "sí", "y", "yes"}:
            rc = _run_data_collection()
            if rc != 0:
                print(f"\n[Nota] data_collection terminó con código {rc}.")
            while True:
                decision = _after_capture_decision(base)
                if decision == "again":
                    rc = _run_data_collection()
                    if rc != 0:
                        print(f"\n[Nota] data_collection terminó con código {rc}.")
                    continue
                if decision == "split":
                    return "continue"
                if decision == "restart":
                    return "restart"
                sys.exit(0)

        print("\n→ Se usarán las imágenes ya guardadas en my_images (sin nueva captura).")
        return "continue"


def main() -> None:
    print("\n=== assistant_flow.py (modo guiado) ===")
    print("Este flujo automatiza split + preprocessing + entrenamiento por camino.")

    while True:
        path_num = _ask_path()
        source = _ask_source(path_num)
        _write_data_source(source)

        if source == "my_images":
            flow = _handle_my_images_flow()
            if flow == "restart":
                print("\n>>> Reiniciando asistente (vuelves a elegir camino)...\n")
                continue
        _ensure_source_exists(source)

        print("\nEjecutando flujo guiado completo (split → preprocesado → entrenamiento)...")
        try:
            _run([sys.executable, str(SCRIPTS_DIR / "data_split.py")])
            _run([sys.executable, str(SCRIPTS_DIR / "data_preprocessing.py")])
            _run([sys.executable, str(SCRIPTS_DIR / "run_model_path.py"), "--path", str(path_num)])
        except RuntimeError as e:
            print(f"\n❌ Error en el pipeline: {e}")
            retry = input(
                "\n¿Volver al inicio del asistente? (S/n): "
            ).strip().lower()
            if retry in {"", "s", "si", "sí", "y", "yes"}:
                continue
            raise
        break

    print("\n✅ Flujo guiado terminado.")


if __name__ == "__main__":
    main()
