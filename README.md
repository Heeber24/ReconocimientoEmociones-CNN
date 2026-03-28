# Reconocimiento de emociones con redes neuronales (CNN)

## Bienvenida

Este trabajo tiene como **objetivo** servir de **ejemplo práctico** de cómo abordar un problema de visión por computadora con **redes neuronales convolucionales**: capturar o reunir imágenes, prepararlas, dividirlas en entrenamiento / validación / prueba, entrenar distintos modelos (CNN desde cero, transfer learning con EfficientNet, transfer desde otro modelo ya entrenado) y probar el resultado **en vivo** con la cámara.

No sustituye un curso completo de aprendizaje profundo, pero muestra un **flujo ordenado** que puedes seguir, modificar y documentar para clase o entrega.

**Importante:** la selección de dataset se centraliza en `scripts/project_config.py` con `DATA_SOURCE` (`fer_2013`, `affectnet`, `my_images`), y la reutilizan `data_split.py`, `data_preprocessing.py` y `realtime_emotion_recognition.py`.

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

**`project_config.py`**  
Configuración **global** del origen de datos: variable **`DATA_SOURCE`**, con valor **`"my_images"`**, **`"fer_2013"`** o **`"affectnet"`**. Apunta a las carpetas bajo `data/` que usan `data_split.py`, `data_preprocessing.py` y la lógica de previsualización en `realtime_emotion_recognition.py`. **Cámbiala una sola vez** antes de split/preprocess/entrenar para que todo el pipeline sea coherente.

**`quiet_console.py`**  
Módulo **compartido** (no lo borres): reduce **warnings** y **logs** de TensorFlow, absl, OpenCV, etc. Lo importan casi todos los scripts al inicio (`init()`; y tras cargar TF, `silence_tensorflow_post_import()` donde aplica). Sirve para ver la consola más limpia al entrenar y al usar la cámara.

**`data_collection.py`**  
Captura rostros con **OpenCV** y **webcam**. Recortes **224×224** en color (**BGR** al disco; el preprocesado pasa a **RGB** para el modelo). Menú interactivo: **emoción** y **cámara** (0 interna / 1 externa) con la **misma lógica de swap** que `realtime_emotion_recognition.py` (`SWAP_CAMERA_INDICES_0_AND_1`). Vista tipo **realtime**: fondo difuminado y rostro nítido (solo visual; **las imágenes guardadas siguen siendo nítidas**). Marco verde explícito de **“zona que se guarda”** mientras capturas. Texto de ayuda en **azul rey** (BGR). **Cierra con la X** de la ventana (no se insiste en ESC); si cierras **mientras capturas**, **tkinter** pregunta si seguro. Tras cerrar, el **asistente guiado** (`assistant_flow.py`) ya no enchaina el split solo: pide confirmación (ver abajo).

**`data_split.py`**  
Lee el origen según **`DATA_SOURCE`** (`data/my_images`, `data/FER_2013` o `data/AffectNet`) y reparte en **train / validation / test** bajo **`data/prepared_data/`**. **Borra** la carpeta `prepared_data` anterior antes de copiar. Clases = subcarpetas por emoción (`angry`, `happy`, `neutral`, `surprise`).

**`data_preprocessing.py`**  
Define **normalización**, **data augmentation** y **generadores** `ImageDataGenerator` usados por `training_utils.py`. Si **`DATA_SOURCE == "my_images"`**, convierte **BGR→RGB** al leer (OpenCV guarda BGR). Al ejecutarlo como script valida rutas y conteos.

**`training_utils.py`**  
Lógica de **entrenamiento**: CNN desde cero, **EfficientNetB0** (ImageNet) y **transfer** desde modelo guardado (caminos 5 y 6). **Callbacks**, evaluación en **test** y rutas fijas **`models/modelo_camino_1.keras` … `modelo_camino_6.keras`** (más copias etiquetadas con fecha/accuracy).

**`run_model_path.py`**  
**Único punto de entrada** para entrenar un camino **1…6**: menú interactivo o **`python scripts/run_model_path.py --path N`**. Valida que **`DATA_SOURCE`** en `project_config.py` coincida con lo que exige ese camino (propias vs FER/AffectNet). En **5 y 6** pide el **modelo base** (`MODELO_BASE`) y avisa si no cumple la regla pedagógica del curso. **No** ejecuta split ni preprocess por ti: debes haber corrido **`data_split.py`** y **`data_preprocessing.py`** antes (o usar `assistant_flow.py`).

**`assistant_flow.py`**  
**Modo guiado** para alumnos: elige camino → ajusta `DATA_SOURCE` en disco → si toca **`my_images`**, opción de **capturar**. **Importante:** cada vez que termina **`data_collection.py`**, muestra un **menú obligatorio**: capturar de nuevo, **continuar con split** usando lo que hay en disco, **reiniciar el asistente** desde el inicio o salir. Así **no** se lanza el split **en automático** al cerrar la ventana sin haber confirmado. Luego ejecuta en cadena: **`data_split.py`** → **`data_preprocessing.py`** → **`run_model_path.py --path N`**. Si falla un paso, ofrece **volver al inicio**.

**`realtime_emotion_recognition.py`**  
Reconocimiento **en vivo**: Haar Cascade, ROI según **`TRAINED_DATA_SOURCE`** (derivado de `DATA_SOURCE`: gris+RGB para FER, color para AffectNet/my_images con BGR si aplica). Carga **`.keras`** con compatibilidad (`custom_objects`, `Dense` sin `quantization_config` problemático). **`--model-path`**, **`--camera`**, **`--no-swap-camera`**, selector opcional de **`modelo_camino_N.keras`** en terminal (`SELECT_MODEL_FROM_TERMINAL`), etiqueta en pantalla del modelo en uso, panel de probabilidades, fondo difuminado. Ventana con **X** para cerrar (comportamiento tipo `WND_PROP_VISIBLE`). Consola atenuada vía **`quiet_console`**.

**`bibliotecas_versiones.py`**  
Comprueba dependencias; al ejecutarlo usa también **`quiet_console`** al inicio de `main()`.

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
│   ├── project_config.py      (DATA_SOURCE global)
│   ├── quiet_console.py       (warnings / logs; no eliminar)
│   ├── data_collection.py
│   ├── data_split.py
│   ├── data_preprocessing.py
│   ├── training_utils.py
│   ├── run_model_path.py      (entrenar camino 1–6)
│   ├── assistant_flow.py      (flujo guiado alumnos)
│   ├── realtime_emotion_recognition.py
│   └── bibliotecas_versiones.py
├── data/
│   ├── my_images/             (captura propia)
│   ├── FER_2013/              (dataset base 1)
│   ├── AffectNet/             (dataset base 2)
│   └── prepared_data/         (train, validation, test)
├── models/                    (modelo_camino_1.keras … modelo_camino_6.keras)
├── requirements.txt
├── README.md
└── NOTAS_TECNICAS.txt
```

---

## 6. Flujo típico (orden de ejecución)

### 6.1 Manual (control total)

1. **`DATA_SOURCE`** en `scripts/project_config.py` (`my_images`, `fer_2013`, `affectnet`).
2. Si usas **datos propios**: `python scripts/data_collection.py` (tantas veces como necesites).
3. `python scripts/data_split.py`
4. `python scripts/data_preprocessing.py`
5. Entrenar un camino: **`python scripts/run_model_path.py --path N`** (N = 1…6) o `python scripts/run_model_path.py` y elige en el menú.
6. Prueba en cámara: `python scripts/realtime_emotion_recognition.py` (y/o `--model-path`, `--camera`).

**`run_model_path.py` no corre split ni preprocess** por sí solo: el orden 3 → 4 → 5 es obligatorio antes de entrenar.

### 6.2 Modo guiado (recomendado si empiezas)

```text
python scripts/assistant_flow.py
```

Te lleva por camino 1–6, fuente de datos, captura opcional **`my_images`**, y **después de cada sesión de `data_collection`** debes elegir explícitamente si **continuar al split** o **volver a capturar / reiniciar**. Luego encadena split → preprocess → `run_model_path.py`.

**Caminos 5 y 6:** necesitas **`modelo_camino_3.keras`** (5) o **`modelo_camino_1.keras`** (6); el script te pide ruta de base y valida coherencia con `DATA_SOURCE`.

---

## 7. Los seis caminos (idea de cada uno)

1. **Camino 1** — CNN desde cero, **tus** imágenes.  
2. **Camino 2** — Mismos datos que el 1, **EfficientNetB0** (ImageNet).  
3. **Camino 3** — CNN desde cero, dataset base (**FER_2013** o **AffectNet**).  
4. **Camino 4** — EfficientNet con dataset base (**FER_2013** o **AffectNet**).  
5. **Camino 5** — Transfer desde el modelo del **3**, entrenando con **tus** fotos.  
6. **Camino 6** — Transfer desde el modelo del **1**, entrenando con dataset base.

Cada uno guarda su archivo principal en **`models/modelo_camino_N.keras`** y una copia con fecha y accuracy.

---

## 8. Dónde configurar (resumen)

- Origen de datos (global): **`scripts/project_config.py`** (`DATA_SOURCE`).
- Nombres de los seis checkpoints: **`training_utils.py`** (`MODEL_CAMINO_1` … `MODEL_CAMINO_6`).
- Qué camino entrenar: **`run_model_path.py`** (`--path N` o menú). **Ya no** existen `generate_model_path_1.py` … `6.py`.
- Base en caminos **5 y 6**: **`MODELO_BASE`** (y prompts) dentro de **`run_model_path.py`** al ejecutarlo.
- Cámara y modelo en vivo: **`realtime_emotion_recognition.py`** (`MODELO_REALTIME`, `SELECT_MODEL_FROM_TERMINAL`, `INDICE_MODELO`, `CANDIDATOS_EN_MODELS`, **`TRAINED_DATA_SOURCE`** / `DATA_SOURCE`, flags **`--model-path`**, **`--camera`**, **`--no-swap-camera`**).
- Captura: **`data_collection.py`** (`SWAP_CAMERA_INDICES_0_AND_1`, `BLUR_BACKGROUND`, `MAX_IMAGES`, etc.).
- Consola silenciosa: **`quiet_console.py`** (no borrar; lo usan los demás scripts).

---

## 9. Realtime

- Prioridad: argumento **`--model-path`**.
- Si no: con **`SELECT_MODEL_FROM_TERMINAL = True`** puedes elegir en consola un **`modelo_camino_N.keras`** estándar; si no, **`MODELO_REALTIME`** o **`INDICE_MODELO`** + **`CANDIDATOS_EN_MODELS`**.
- **`TRAINED_DATA_SOURCE`** sigue a **`DATA_SOURCE`** de `project_config.py` para el preprocesado del ROI (FER gris→RGB, etc.).
- Cierra la ventana con la **X** (misma idea que en captura). TensorFlow/OpenCV suelen ir más callados gracias a **`quiet_console`**.

---

## 10. Transfer desde ImageNet

En código está fijado **EfficientNetB0** (no VGG16). Otro backbone implica editar `training_utils.py` y la carga en `realtime_emotion_recognition.py`.

---

## 11. Subir al repositorio (Git), Colab con datasets base y qué esperar del rendimiento

### 11.1 Antes de hacer `push` al repo

Revisa el archivo **`.gitignore`** en la raíz del proyecto. En esta plantilla suele estar ignorado, entre otras cosas:

- **`venv/`** — el entorno virtual no debe subirse (cada quien lo crea en su máquina con `python -m venv`).
- **`*.keras`** y **`*.h5`** — los modelos entrenados pesan mucho; no conviene llenar el repositorio con ellos.
- **`data/`** — muchas veces se ignora toda la carpeta para no subir miles de imágenes ni datos personales.

Eso implica: el que clone el repo tendrá el **código y la documentación**, pero tendrá que **colocar sus propias imágenes** (en `data/my_images`) o un dataset base (en `data/FER_2013` o `data/AffectNet`) y volver a entrenar. Si tu profesor pide entregar también datos o un modelo, usa **Drive**, **releases** o un zip aparte según las reglas de la asignatura.

### 11.2 Probar en Google Colab con FER_2013 o AffectNet

Objetivo típico: entrenar con un **dataset base** (FER_2013 o AffectNet, con muchas imágenes y opcionalmente **GPU** en Colab) y comparar con lo que obtuviste con **tus fotos** en local.

Enlace FER_2013 listo para clase (cuatro emociones):  
<https://drive.google.com/drive/folders/1KskRFbO7H2qRqh3A0vtUXe-loXfivuQy?usp=sharing>

Enlace AffectNet listo para clase (cuatro emociones):  
<https://drive.google.com/drive/folders/1U03VkNE9UVSIe2YlgdwhZ39utDarJoj9?usp=sharing>

Pasos alineados con este proyecto:

1. Sube el proyecto a tu repo y/o empaqueta un **zip** con al menos `scripts/`, `requirements.txt`, `README.md`, `NOTAS_TECNICAS.txt` (y sin `venv` ni modelos gigantes si no quieres).
2. En **Google Drive**, deja el dataset en la ruta que espera el código:
   - FER_2013: **`data/FER_2013/`**
   - AffectNet: **`data/AffectNet/`**
   Con **subcarpetas por emoción** y los mismos nombres que usa el proyecto (`angry`, `happy`, `neutral`, `surprise`).
3. En Colab: monta Drive, descomprime o clona el proyecto, `cd` a la raíz (donde está `requirements.txt`). Guía detallada celda por celda: **`NOTAS_TECNICAS.txt`**, sección **6**.
4. En **`scripts/project_config.py`** define **`DATA_SOURCE = "fer_2013"`** o **`"affectnet"`** según el dataset que vayas a usar.
5. Ejecuta **`data_split.py`** y **`data_preprocessing.py`**.
6. Entrena con **`python scripts/run_model_path.py --path 3`** (CNN + dataset base) o **`--path 4`** (EfficientNet + dataset base).
7. Copia **`modelo_camino_3.keras`** o **`modelo_camino_4.keras`** a Drive o **descárgalo** a tu PC para probarlo con **`realtime_emotion_recognition.py`** (ajusta **`DATA_SOURCE`** / **`TRAINED_DATA_SOURCE`** acorde al modelo y al dominio).

### 11.3 ¿Va a salir “tan bueno” como con tus imágenes?

**No está garantizado** que el número de precisión en test sea mayor o menor: **dataset base y tus fotos no son el mismo mundo**.

- FER_2013/AffectNet pueden tener otro tipo de rostros, resolución, iluminación y recorte que lo que capturas tú con la webcam.
- Un modelo puede ir **muy bien** en el test de FER y **regular** contigo en vivo, o al revés.
- Por eso el proyecto tiene **varios caminos** (1 frente a 3, 2 frente a 4, y los transfer 5 y 6): para **comparar** estrategias y dominios, no solo un único “mejor modelo universal”.

En la práctica: usa la **métrica en test** del script como referencia honesta para ese dataset, y la **cámara** como prueba en **tu** dominio.

---

## 12. Más detalle técnico

**`NOTAS_TECNICAS.txt`** — venv desde cero, Colab paso a paso, GPU local, orden con **`run_model_path.py`**, sección **9** (`assistant_flow`, captura, `quiet_console`), git/archivos pesados (sección **10**).

**`TEORIA_REDES_NEURONALES.md`** — teoría de entrenamiento (coste, gradiente, backprop), **CNN**, **data augmentation**, **transfer learning**, **frameworks** (TensorFlow, Keras, PyTorch, JAX) y **aspectos prácticos** (activaciones, BatchNorm, regularización), enlazados al código del proyecto.

## 13. Enlace de colab
https://colab.research.google.com/drive/1BvtUT08zLA-mb-Pazveou2R1PpEZakpN?authuser=1#scrollTo=ZWA76CLUmKnS

## 14. Presentación 
https://drive.google.com/file/d/1pzUJNWNmUYlu74NvGXl0bgRrmHj7Y9Ps/view?usp=sharing
