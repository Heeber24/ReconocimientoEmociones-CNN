# Ejecutar el proyecto en Google Colab – Paso a paso

Sigue estos pasos en orden. Puedes copiar cada bloque en una **celda** del notebook.

---

## Paso 1: Abrir Colab y clonar el repositorio

1. Entra en [Google Colab](https://colab.research.google.com).
2. **File → New notebook** (nuevo notebook).
3. En la primera celda, ejecuta:

```python
# Clonar tu repositorio (reemplaza con tu usuario si es otro)
!git clone https://github.com/Heeber24/ReconocimientoEmociones-CNN.git
%cd ReconocimientoEmociones-CNN
```

Así el proyecto queda en Colab y el directorio de trabajo es la raíz del repo.

---

## Paso 2: Instalar dependencias

En una nueva celda:

```python
!pip install -r requirements.txt -q
```

El `-q` reduce el texto en pantalla. Espera a que termine.

---

## Paso 3: Subir el dataset (elegir una opción)

El código espera las imágenes en **data/kaggle_fer/** con carpetas **angry**, **happy**, **neutral**, **surprise**.

### Opción A: Tienes FER2013 en Google Drive

1. Sube el dataset a Drive (por ejemplo en `Mi unidad/fer2013/` con las 4 carpetas).
2. En una celda:

```python
from google.colab import drive
drive.mount('/content/drive')

# Crear carpeta y copiar (ajusta la ruta a donde esté tu dataset en Drive)
!mkdir -p data/kaggle_fer
!cp -r "/content/drive/MyDrive/fer2013/"* data/kaggle_fer/
# Si tu estructura en Drive ya es fer2013/angry, fer2013/happy, etc., el comando anterior vale.
# Si está en otra ruta, cambia "/content/drive/MyDrive/fer2013/" por tu ruta.
```

### Opción B: Descargar FER desde Kaggle en Colab

1. En [Kaggle](https://www.kaggle.com) → Account → Create New API Token (descarga `kaggle.json`).
2. Sube `kaggle.json` a Colab o ejecuta y sube cuando pida:

```python
# Sube tu archivo kaggle.json cuando lo pida
from google.colab import files
uploaded = files.upload()  # Elige kaggle.json

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Descargar dataset (busca el nombre exacto en Kaggle, ej. "msambare/fer2013")
!kaggle datasets download -d msambare/fer2013 --unzip
!mkdir -p data/kaggle_fer
# Mueve las carpetas al lugar esperado (ajusta según cómo venga el ZIP)
# !mv fer2013/train/angry data/kaggle_fer/  # repite para happy, neutral, surprise
```

(Ajusta nombres de carpetas según el dataset que bajes.)

### Opción C: Probar sin dataset completo (solo estructura)

Si solo quieres ver que el flujo corre sin entrenar de verdad:

```python
import os
for em in ["angry", "happy", "neutral", "surprise"]:
    os.makedirs(f"data/kaggle_fer/{em}", exist_ok=True)
# Las carpetas quedan vacías; data_split y preprocesado correrán, pero no hay imágenes para entrenar bien.
```

---

## Paso 4: Comprobar que los datos están en su sitio

```python
!ls data/kaggle_fer/
!python scripts/bibliotecas_versiones.py
```

Deberías ver las 4 carpetas y un mensaje de que las librerías están bien.

---

## Paso 5: Elegir flujo de entrenamiento

### Opción recomendada para clase (máxima precisión): Transfer Learning

Usa FER, hace split, preprocesado y entrena con **Transfer Learning** (EfficientNetB0). El modelo suele dar mejor precisión y luego lo bajas para correr el reconocedor en tu laptop.

```python
!python scripts/run_flujo_transfer.py
```

Eso ejecuta: **data_split** → **data_preprocessing** → **train_transfer_imagenet.py**.  
Al terminar, el modelo queda en `models/emotion_recognition_EfficientNetB0_model.keras`.

Para usarlo en tu PC: descarga ese archivo (o cópialo a Drive), ponlo en `models/` de tu proyecto local y en `realtime_emotion_recognition.py` pon **use_custom_cnn = False** y **TRANSFER_MODEL_NAME = "EfficientNetB0"**.

### Opción alternativa: CNN desde cero

```python
!python scripts/run_flujo_completo.py
```

Ejecuta: data_split → data_preprocessing → **train_cnn_from_scratch.py** (CNN desde cero, sin transfer).  
En realtime usarías **use_custom_cnn = True**.

---

## Paso 6: Usar la GPU en Colab (recomendado)

Antes de entrenar:

- **Runtime → Change runtime type → Hardware accelerator → T4 GPU** (o la que ofrezca Colab).
- Luego ejecuta de nuevo la celda del Paso 5 si ya la habías corrido sin GPU.

---

## Paso 7: Probar el modelo en tiempo real (realtime)

En Colab la cámara del portátil no se usa igual que en local. Opciones:

- **En tu PC:** Descarga el modelo desde Colab (ver Paso 8) y ejecuta en tu máquina:
  ```bash
  python scripts/realtime_emotion_recognition.py
  ```
- **En Colab con la cámara del navegador:** Se puede hacer con código extra (por ejemplo `javascript` para capturar video). Si quieres, en otro documento te dejo un ejemplo de celda para realtime en Colab.

---

## Paso 8: Descargar el modelo entrenado

Para llevártelo a tu PC:

```python
from google.colab import files
# Si usaste Transfer Learning (run_flujo_transfer.py):
files.download('models/emotion_recognition_EfficientNetB0_model.keras')
# Si usaste CNN desde cero (run_flujo_completo.py):
# files.download('models/emotion_recognition_Personal.keras')
```

Guarda el archivo en la carpeta **models/** de tu proyecto local. En el repo los `.keras` están en `.gitignore`, así que **no se suben con git push**; para usarlo en tu laptop usa descarga desde Colab o cópialo desde Drive.

---

## Resumen rápido (orden de celdas)

| # | Qué hacer |
|---|-----------|
| 1 | `!git clone https://github.com/Heeber24/ReconocimientoEmociones-CNN.git` y `%cd ReconocimientoEmociones-CNN` |
| 2 | `!pip install -r requirements.txt -q` |
| 3 | Montar Drive y copiar dataset a `data/kaggle_fer/` (o descargar con Kaggle / crear carpetas vacías) |
| 4 | `!ls data/kaggle_fer/` y `!python scripts/bibliotecas_versiones.py` |
| 5 | **Runtime → Change runtime type → GPU** |
| 6 | `!python scripts/run_flujo_transfer.py` (transfer) o `!python scripts/run_flujo_completo.py` (CNN desde cero) |
| 7 | Descargar `.keras` con `files.download('models/emotion_recognition_EfficientNetB0_model.keras')` y en local: realtime con **use_custom_cnn = False** |

Si algo falla, revisa que la ruta de datos sea `data/kaggle_fer/` con las cuatro carpetas **angry**, **happy**, **neutral**, **surprise** y que en `scripts/config.py` tengas `USE_KAGGLE_FER = True`.
