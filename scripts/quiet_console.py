# -*- coding: utf-8 -*-
"""
Reduce warnings y logs ruidosos en consola (TensorFlow, absl, h5py, etc.).

Uso en cada script ejecutable, lo antes posible (antes de importar tensorflow):

    import quiet_console
    quiet_console.init()

Si usas OpenCV, después de ``import cv2``:

    quiet_console.silence_opencv()

Después de importar TensorFlow / Keras (una vez cargado el módulo):

    quiet_console.silence_tensorflow_post_import()
"""
from __future__ import annotations

import logging
import os
import warnings

_INITIALIZED = False


def init() -> None:
    """Idempotente: seguro llamar varias veces."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    # Antes de importar tensorflow (mensajes C++, oneDNN, etc.)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    # Evita el aviso habitual de "oneDNN custom operations" en CPU Intel (puedes poner 1 en el entorno si quieres oneDNN).
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    warnings.simplefilter("ignore")
    warnings.filterwarnings("ignore")
    for cat in (
        DeprecationWarning,
        FutureWarning,
        UserWarning,
        PendingDeprecationWarning,
    ):
        warnings.filterwarnings("ignore", category=cat)

    for name in (
        "tensorflow",
        "tf_keras",
        "keras",
        "keras.src",
        "absl",
        "h5py",
        "matplotlib",
        "PIL",
        "urllib3",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)

    try:
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
    except Exception:
        pass


def silence_tensorflow_post_import() -> None:
    """
    TensorFlow/Keras registran loggers al importar; conviene bajar el nivel
    después de ``import tensorflow`` (no solo al inicio del proceso).
    """
    try:
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("tensorflow.core").setLevel(logging.ERROR)
        logging.getLogger("tf_keras").setLevel(logging.ERROR)
        logging.getLogger("keras").setLevel(logging.ERROR)

        try:
            tf.autograph.set_verbosity(0)
        except Exception:
            pass
        try:
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        except Exception:
            pass
    except Exception:
        pass


def silence_opencv() -> None:
    try:
        import cv2

        cv2.setLogLevel(0)
    except Exception:
        pass
