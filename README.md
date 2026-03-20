# Reconocimiento de emociones con redes neuronales (CNN)

## Bienvenida

Este trabajo tiene como **objetivo** servir de **ejemplo práctico** de cómo abordar un problema de visión por computadora con **redes neuronales convolucionales**: capturar o reunir imágenes, prepararlas, dividirlas en entrenamiento / validación / prueba, entrenar distintos modelos (CNN desde cero, transfer learning con EfficientNet, transfer desde otro modelo ya entrenado) y probar el resultado **en vivo** con la cámara.

No sustituye un curso completo de aprendizaje profundo, pero muestra un **flujo ordenado** que puedes seguir, modificar y documentar para clase o entrega.

**Importante:** no hay un `config.py` central. La configuración va en bloques **CONFIGURA** al inicio de cada script (`USE_KAGGLE_FER`, rutas, modelo en realtime, etc.).

---

## 1. Librerías necesarias

### 1.1 Lista oficial

Todas las versiones mínimas y paquetes extra están en **`requirements.txt`**.

**Entorno virtual:** conviene crear un `venv`, activarlo y **después** instalar dependencias (comandos exactos para Windows y Linux/mac en **`NOTAS_TECNICAS.txt`**, sección 1).

Con el venv **activado** y desde la raíz del proyecto:

```text
pip install -r requirements.txt
```

**Paso a paso sin dar nada por sabido:** en **`NOTAS_TECNICAS.txt`** tienes sección **1** (qué es el venv, comprobar Python, `cd` al proyecto, crear, activar, `pip`, errores frecuentes) y sección **6** (Google Colab: subir zip, GPU, montar Drive, descomprimir, `pip`, correr scripts, bajar `.keras`, limitaciones).

### 1.2 Script para probar si tienes todo instalado

En la raíz del proyecto ejecuta:

```text
python scripts/bibliotecas_versiones.py
```

Ese script intenta importar TensorFlow, NumPy, OpenCV, scikit-learn, Pillow, SciPy, Matplotlib y confirma que `shutil` está disponible (es parte de la librería estándar de Python, **no** se instala con pip).

### 1.3 Código de prueba rápida (copiar y pegar)

Si prefieres probar a mano en un intérprete o en un `.py` vacío, puedes usar algo equivalente a esto. **Nota:** en este proyecto la API de Keras que usamos es la que viene **dentro de TensorFlow** (`tf.keras`), no hace falta un paquete `keras` aparte si ya instalaste TensorFlow.

```python
import tensorflow as tf  # pip install tensorflow
import numpy as np  # pip install numpy
import cv2  # pip install opencv-python
import sklearn  # pip install scikit-learn
import shutil  # módulo estándar de Python, no uses pip para shutil

print("TensorFlow:", tf.__version__)
print("Keras (tf.keras):", tf.keras.__version__)
print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)
print("scikit-learn:", sklearn.__version__)
print("shutil: OK (viene con Python, no instalar con pip)")
```

Opcionalmente, si `requirements.txt` está completo:

```python
from PIL import Image  # pip install Pillow
import scipy  # pip install scipy
import matplotlib  # pip install matplotlib
```

---

## 2. Distribución de datos (train, validation, test)

En aprendizaje automático conviene separar los datos en subconjuntos con **roles distintos**:

**a) Train (entrenamiento)**

- **Propósito:** el modelo usa estas imágenes para **aprender** (actualizar pesos).
- **Cantidad típica en textos:** suele hablarse del **70–80 %** del total.
- **Ejemplo ilustrativo:** si tienes unas 800 imágenes en total (200 por emoción), podrías usar del orden de **140–160 imágenes por emoción** para entrenamiento.

**b) Validation (validación)**

- **Propósito:** durante el entrenamiento sirve para **ajustar el comportamiento del entrenamiento** (por ejemplo early stopping, guardar el mejor modelo, reducir learning rate) **sin** usar para el gradiente de entrenamiento. Así reduces el sesgo de evaluar solo con lo que ya memorizó el modelo.
- **Cantidad típica:** **10–15 %** del total.
- **Ejemplo:** **20–30 imágenes por emoción** para validación.

**c) Test (prueba)**

- **Propósito:** **evaluación final** después de entrenar, para una métrica más objetiva. En este proyecto, al terminar el entrenamiento se llama a `evaluate` sobre **test**; la precisión que ves y la que va en el nombre de las copias etiquetadas sale de ahí.
- **Cantidad típica:** otro **10–15 %**.
- **Ejemplo:** **20–30 imágenes por emoción** reservadas solo para test.

**Qué hace exactamente este repositorio en `data_split.py`**

Las proporciones están fijadas en código: **70 % train**, **15 % validation**, **15 % test** (aproximado, con estratificación por clase). Puedes leer los comentarios al inicio de `scripts/data_split.py` si quieres cambiar porcentajes.

---

## 3. Descripción detallada de los scripts

**`data_collection.py`**  
Utiliza **OpenCV** para capturar rostros en tiempo real con la **webcam**. Los recortes se redimensionan a **224×224 píxeles en color** (BGR al guardar; en el preprocesado se pasa a RGB para el modelo). Cada imagen se guarda en una **carpeta** según la emoción elegida (`angry`, `happy`, `neutral`, `surprise`). El usuario elige emoción por menú y puede limitar cuántas imágenes captura por sesión. Es el paso opcional cuando trabajas con **datos propios** y no solo con FER.

**`data_split.py`**  
Organiza las imágenes del origen (`data/images/data_collection` o `data/kaggle_fer`, según `USE_KAGGLE_FER`) en tres subconjuntos: **entrenamiento**, **validación** y **prueba**. Crea la estructura bajo `data/images/prepared_data/train`, `validation` y `test`, con subcarpetas por emoción. **Antes de repartir, borra** la carpeta `prepared_data` anterior para no mezclar corridas viejas. Es un paso estándar en flujos de visión por computadora.

**`data_preprocessing.py`**  
Prepara la **lectura** de las imágenes para el entrenamiento: **normalización** (por ejemplo escala 0–1), **augmentation** en entrenamiento (rotaciones, brillo, etc.) y creación de **generadores** de Keras listos para `model.fit`. Al ejecutarlo como script, valida que existan train/validation, muestra conteos y comprueba que los generadores cargan sin error. El entrenamiento en `training_utils.py` **reutiliza** estas mismas funciones.

**`training_utils.py`**  
Aquí está la **lógica pesada** del entrenamiento: construcción de la CNN desde cero, modelo con **EfficientNetB0** e ImageNet, y transfer **desde un modelo ya guardado** (caminos 5 y 6). Incluye **callbacks** (por ejemplo guardar el mejor modelo por `val_loss` o `val_accuracy`, **EarlyStopping**, **ReduceLROnPlateau** según el caso) y al final **evalúa en el conjunto de prueba**. También define las rutas **`modelo_camino_1.keras` … `modelo_camino_6.keras`**. Las carpetas `train/`, `validation/`, `test/` las toma desde `data_preprocessing.py`.

**`generate_model_path_1.py` … `generate_model_path_6.py`**  
Scripts **cortos** que solo fijan **qué camino** corres (datos propios vs FER, coherente con split/preprocess) y llaman a la función adecuada en `training_utils.py`. Así el flujo pedagógico queda claro: primero datos, luego el número de camino que quieras experimentar.

**`realtime_emotion_recognition.py`**  
Script de **reconocimiento en tiempo real**: abre la cámara, detecta rostros (Haar Cascade), recorta la región, la prepara según el tipo de modelo y pasa el tensor por la **red cargada** (`.keras`). Muestra la emoción predicha y un panel con **probabilidades** por clase sobre el video.

**`bibliotecas_versiones.py`**  
Comprueba imports y versiones de las dependencias principales; úsalo después de `pip install -r requirements.txt`.

---

## 4. Uso de train, validation y test dentro del código (resumen técnico)

1. **Train** — el modelo aprende con estos lotes.  
2. **Validation** — durante `fit`, para callbacks (mejor checkpoint, parada temprana, LR).  
3. **Test** — al final, `model.evaluate` sobre test; esa es la base de la métrica que imprimes y de la copia con `acc…` en el nombre.

---

## 5. Estructura del proyecto

```text
ReconocimientoEmociones-CNN/
├── scripts/
├── data/
│   ├── images/
│   │   ├── data_collection/
│   │   └── prepared_data/     (train, validation, test)
│   └── kaggle_fer/
├── models/                    (modelo_camino_1.keras … modelo_camino_6.keras)
├── requirements.txt
├── README.md
└── NOTAS_TECNICAS.txt
```

---

## 6. Flujo típico (orden de ejecución)

1. Misma **`USE_KAGGLE_FER`** en `data_split.py` y `data_preprocessing.py` (`False` = propias, `True` = FER).  
2. `python scripts/data_split.py`  
3. `python scripts/data_preprocessing.py`  
4. Misma bandera en el `generate_model_path_N.py` que vayas a usar.  
5. `python scripts/generate_model_path_N.py`  
6. Para webcam: configura y ejecuta `python scripts/realtime_emotion_recognition.py` (o `--model-path`).

**Caminos 5 y 6:** necesitas antes **`modelo_camino_3.keras`** (camino 5) o **`modelo_camino_1.keras`** (camino 6), luego cambias datos y vuelves a split + preprocess.

---

## 7. Los seis caminos (idea de cada uno)

1. **Camino 1** — CNN desde cero, **tus** imágenes.  
2. **Camino 2** — Mismos datos que el 1, **EfficientNetB0** (ImageNet).  
3. **Camino 3** — CNN desde cero, datos **FER**.  
4. **Camino 4** — EfficientNet con datos **FER**.  
5. **Camino 5** — Transfer desde el modelo del **3**, entrenando con **tus** fotos.  
6. **Camino 6** — Transfer desde el modelo del **1**, entrenando con **FER**.

Cada uno guarda su archivo principal en **`models/modelo_camino_N.keras`** y una copia con fecha y accuracy.

---

## 8. Dónde configurar (resumen)

- Origen de datos: **`data_split.py`** y **`data_preprocessing.py`** (`USE_KAGGLE_FER`).  
- Nombres de los seis checkpoints: **`training_utils.py`**.  
- Qué camino corres: cada **`generate_model_path_*.py`**.  
- Base en 5 y 6: variable **`MODELO_BASE`** en esos scripts.  
- Cámara: **`realtime_emotion_recognition.py`** (`MODELO_REALTIME`, `INDICE_MODELO`, **`ENTRENADO_CON_FER`**).

---

## 9. Realtime

- Prioridad: **`--model-path`**.  
- Si no: **`MODELO_REALTIME`** o **`INDICE_MODELO`** + lista **`CANDIDATOS_EN_MODELS`**.

---

## 10. Transfer desde ImageNet

En código está fijado **EfficientNetB0** (no VGG16). Otro backbone implica editar `training_utils.py` y la carga en `realtime_emotion_recognition.py`.

---

## 11. Subir al repositorio (Git), Colab con FER y qué esperar del rendimiento

### 11.1 Antes de hacer `push` al repo

Revisa el archivo **`.gitignore`** en la raíz del proyecto. En esta plantilla suele estar ignorado, entre otras cosas:

- **`venv/`** — el entorno virtual no debe subirse (cada quien lo crea en su máquina con `python -m venv`).
- **`*.keras`** y **`*.h5`** — los modelos entrenados pesan mucho; no conviene llenar el repositorio con ellos.
- **`data/`** — muchas veces se ignora toda la carpeta para no subir miles de imágenes ni datos personales.

Eso implica: el que clone el repo tendrá el **código y la documentación**, pero tendrá que **colocar sus propias imágenes** (en `data/images/data_collection` o `data/kaggle_fer` según el caso) y volver a entrenar. Si tu profesor pide entregar también datos o un modelo, usa **Drive**, **releases** o un zip aparte según las reglas de la asignatura.

### 11.2 Probar en Google Colab con datos FER

Objetivo típico: entrenar con **FER** (muchas imágenes, a veces con **GPU** en Colab) y comparar con lo que obtuviste con **tus fotos** en local.

Enlace FER listo para clase (cuatro emociones):  
<https://drive.google.com/drive/folders/1KskRFbO7H2qRqh3A0vtUXe-loXfivuQy?usp=sharing>

Pasos alineados con este proyecto:

1. Sube el proyecto a tu repo y/o empaqueta un **zip** con al menos `scripts/`, `requirements.txt`, `README.md`, `NOTAS_TECNICAS.txt` (y sin `venv` ni modelos gigantes si no quieres).
2. En **Google Drive**, deja el dataset FER en la ruta que espera el código: **`data/kaggle_fer/`**, con **subcarpetas por emoción** y los mismos nombres que usa el proyecto (`angry`, `happy`, `neutral`, `surprise`).
3. En Colab: monta Drive, descomprime o clona el proyecto, `cd` a la raíz (donde está `requirements.txt`). Guía detallada celda por celda: **`NOTAS_TECNICAS.txt`**, sección **6**.
4. En **`data_split.py`** y **`data_preprocessing.py`** pon **`USE_KAGGLE_FER = True`**.
5. Ejecuta **`data_split.py`** y **`data_preprocessing.py`**.
6. Entrena **camino 3** (CNN + FER) o **camino 4** (EfficientNet + FER); en ese `generate_model_path_*.py` debe ir la **misma** `USE_KAGGLE_FER = True`.
7. Copia **`modelo_camino_3.keras`** o **`modelo_camino_4.keras`** a Drive o **descárgalo** a tu PC para probarlo con **`realtime_emotion_recognition.py`** (y ajusta **`ENTRENADO_CON_FER`** acorde al modelo).

### 11.3 ¿Va a salir “tan bueno” como con tus imágenes?

**No está garantizado** que el número de precisión en test sea mayor o menor: **FER y tus fotos no son el mismo mundo**.

- FER tiene otro tipo de rostros, resolución, iluminación y recorte que lo que capturas tú con la webcam.
- Un modelo puede ir **muy bien** en el test de FER y **regular** contigo en vivo, o al revés.
- Por eso el proyecto tiene **varios caminos** (1 frente a 3, 2 frente a 4, y los transfer 5 y 6): para **comparar** estrategias y dominios, no solo un único “mejor modelo universal”.

En la práctica: usa la **métrica en test** del script como referencia honesta para ese dataset, y la **cámara** como prueba en **tu** dominio.

---

## 12. Más detalle técnico

**`NOTAS_TECNICAS.txt`** — venv explicado desde cero, Colab paso a paso, GPU local, git/archivos pesados.
