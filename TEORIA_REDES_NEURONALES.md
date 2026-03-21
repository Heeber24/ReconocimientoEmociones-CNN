# Teoría de redes neuronales y redes convolucionales aplicada a este proyecto

Documento de apoyo académico: **definiciones**, **formulación matemática cuando aplica**, **ejemplos teóricos** y **referencias explícitas al código** del repositorio *ReconocimientoEmociones-CNN*. El objetivo es justificar por qué el diseño del sistema (CNN, transfer learning, preprocesado, frameworks) es coherente con la práctica estándar en visión por computador.

> **Lectura en GitHub:** las fórmulas usan la sintaxis que GitHub renderiza como matemática: **`$...$`** (inline) y **`$$...$$`** (bloque). Si ves texto crudo tipo `( f_\theta` sin renderizar, abre el archivo en **github.com** (vista del repositorio), no solo el texto copiado. Documentación: [Writing mathematical expressions](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions).

---

## Índice temático sugerido

1. [Problema que resuelve el proyecto](#problema-que-resuelve-el-proyecto)  
2. [Función de coste y clasificación multiclase](#función-de-coste-y-clasificación-multiclase)  
3. [Descenso del gradiente y optimizadores](#descenso-del-gradiente-y-optimizadores)  
4. [Retropropagación del error](#retropropagación-del-error)  
5. [Introducción a las redes neuronales convolucionales](#introducción-a-las-redes-neuronales-convolucionales)  
6. [Capas convolucionales](#capas-convolucionales)  
7. [Arquitecturas CNN para visión por computador](#arquitecturas-cnn-para-visión-por-computador)  
8. [Aumento de datos (data augmentation)](#aumento-de-datos-data-augmentation)  
9. [Aprendizaje por transferencia (transfer learning)](#aprendizaje-por-transferencia-transfer-learning)  
10. [Frameworks de aprendizaje profundo](#frameworks-de-aprendizaje-profundo)  
11. [Aspectos prácticos del entrenamiento en redes profundas](#aspectos-prácticos-del-entrenamiento-en-redes-profundas)  
12. [Tabla de correspondencia teoría–código](#tabla-de-correspondencia-teoría–código)  
13. [Síntesis para exposición oral](#síntesis-para-exposición-oral)

---

## Problema que resuelve el proyecto

**Definición (tarea).** Se trata de **clasificación supervisada de imágenes**: cada imagen de rostro pertenece a una de cuatro clases de emoción (`angry`, `happy`, `neutral`, `surprise`). La red debe aprender una función $f_\theta : \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^{4}$ que asigne a cada imagen un vector de probabilidades sobre las clases.

**Ejemplo teórico.** Una imagen de entrada $x$ de tamaño $224 \times 224 \times 3$ pasa por la red; la salida final es un vector $p = (p_1,\ldots,p_4)$ con $p_k \geq 0$ y $\sum_k p_k = 1$ (softmax). La clase predicha es $\arg\max_k p_k$.

**En este proyecto.** Las rutas de datos van de `data/prepared_data/train|validation|test` (generadas por `data_split.py`). El tamaño de entrada al modelo es **224×224×3** (RGB), alineado con EfficientNet y con la captura en `data_collection.py`. Los generadores en `data_preprocessing.py` usan `class_mode="categorical"` (etiquetas one-hot), coherente con `categorical_crossentropy` y softmax.

---

## Función de coste y clasificación multiclase

**Definición.** La **función de coste** (o **pérdida**) mide la discrepancia entre la predicción del modelo y la etiqueta verdadera. El entrenamiento busca parámetros $\theta$ que **minimicen** el valor esperado de esa pérdida sobre los datos.

**Formulación.** En clasificación multiclase con etiquetas one-hot $y \in \{0,1\}^K$ y probabilidades predichas $\hat{p}$ (salida softmax), la **entropía cruzada categórica** es:

$$
L_{\text{CE}} = - \sum_{k=1}^{K} y_k \log(\hat{p}_k + \varepsilon)
$$

donde $\varepsilon$ es un pequeño valor numérico para estabilidad. Si la clase verdadera es $c$, solo el término $-\log(\hat{p}_c)$ contribuye: penaliza fuerte cuando la probabilidad asignada a la clase correcta es baja.

**Ejemplo teórico.** Si la clase es “happy” y el modelo predice $\hat{p}_{\text{happy}} = 0.9$, el coste es $-\log(0.9) \approx 0.105$. Si predice $\hat{p}_{\text{happy}} = 0.1$, el coste es $-\log(0.1) \approx 2.3$, mucho mayor.

**En este proyecto.** Todos los modelos se compilan con:

```python
loss="categorical_crossentropy"
metrics=["accuracy"]
```

**Ubicación:** `scripts/training_utils.py` en cada `model.compile(...)` (por ejemplo entorno a las líneas que definen CNN desde cero, EfficientNet y transfer desde modelo existente).

**Justificación.** Cuatro clases, etiquetas en formato one-hot desde `ImageDataGenerator(..., class_mode="categorical")` en `data_preprocessing.py`. La última capa usa `activation="softmax"`, que es la pareja estándar de esta función de pérdida.

---

## Descenso del gradiente y optimizadores

**Definición.** El **descenso del gradiente** actualiza los parámetros $\theta$ en la dirección opuesta al gradiente de la pérdida $L$ respecto a $\theta$:

$$
\theta \leftarrow \theta - \alpha \nabla_\theta L
$$

donde $\alpha > 0$ es la **tasa de aprendizaje** (*learning rate*). En la práctica se usa el gradiente estimado sobre un **minilote** (*minibatch*) de ejemplos, no sobre todo el dataset (**descenso estocástico por minilotes**).

**Optimizador Adam (idea).** **Adam** adapta tasas de aprendizaje efectivas por parámetro usando estimaciones de momentos de primer y segundo orden del gradiente. Sigue siendo un método de optimización basado en gradientes; suele converger con menos ajuste manual que el GD puro.

**En este proyecto.**

```python
optimizer=tf.keras.optimizers.Adam(1e-4)   # transfer EfficientNet (cabecera)
optimizer=tf.keras.optimizers.Adam(1e-5)   # fine-tuning (base descongelada parcialmente)
optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005)  # CNN desde cero
```

Además, el callback **`ReduceLROnPlateau`** reduce el learning rate cuando `val_loss` deja de mejorar:

```python
ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=5, min_lr=1e-6)
```

**Ubicación:** `scripts/training_utils.py` (compilación del modelo y lista `callbacks` en `model.fit`).

**Justificación.** Tasas distintas para fase inicial y **fine-tuning** (1e-5) evitan “romper” pesos ya útiles de ImageNet al ajustar la base convolucional.

---

## Retropropagación del error

**Definición.** La **retropropagación** (*backpropagation*) es el algoritmo que calcula $\nabla_\theta L$ para todas las capas de la red aplicando la **regla de la cadena** de forma sistemática desde la capa de salida hacia la entrada. Sin backprop, no se podrían entrenar redes profundas de forma eficiente.

**Flujo conceptual.** (1) *Forward pass:* entrada → salida y pérdida $L$. (2) *Backward pass:* derivadas $\partial L / \partial \theta$ capa a capa. (3) El optimizador actualiza $\theta$.

**En este proyecto.** No se implementa backprop a mano. Al ejecutar `model.fit(...)` en `training_utils.py`, **TensorFlow** construye el grafo de operaciones y aplica diferenciación automática para obtener gradientes y actualizar pesos con Adam.

**Ubicación:** cualquier llamada a `model.fit(...)` en `scripts/training_utils.py`.

**Justificación pedagógica.** El proyecto ilustra la **separación de responsabilidades**: el investigador define arquitectura, pérdida y optimizador; el framework ejecuta el cálculo de gradientes de forma fiable y optimizada (incluido uso de GPU en entornos adecuados).

---

## Introducción a las redes neuronales convolucionales

**Definición.** Una **red neuronal convolucional (CNN)** es una arquitectura diseñada para datos con **estructura espacial** (imágenes, mapas). En lugar de conectar cada píxel con todas las neuronas siguientes (como en una capa densa), se usan **filtros convolucionales** que se deslizan sobre la entrada, reutilizando los mismos pesos en distintas posiciones (**compartición de pesos**).

**Motivación teórica.**

- **Localidad:** patrones relevantes (bordes, texturas) son locales en el espacio.
- **Invariancia aproximada a traslaciones:** el mismo filtro detecta un patrón en distintas posiciones.
- **Reducción de parámetros:** frente a una capa fully connected sobre $224 \times 224 \times 3$ píxeles, una convolución con pocos kernels mantiene el número de parámetros manejable.

**En este proyecto.** El **camino 1 y 3** usan una CNN construida con bloques `Conv2D` → `BatchNormalization` → `MaxPooling2D` → `Dropout`, seguidos de capas densas. Los **caminos 2 y 4** usan **EfficientNetB0** como extractor de características (CNN preentrenada). La inferencia en cámara (`realtime_emotion_recognition.py`) consume el mismo tipo de tensores $224 \times 224 \times 3$.

---

## Capas convolucionales

**Definición.** Una capa **Conv2D** aplica varios **kernels** (filtros) bidimensionales a cada canal espacial. Para cada posición válida del kernel sobre el mapa de entrada, se calcula la suma de productos entre pesos del kernel y valores de entrada (más sesgo), seguido habitualmente de una función de activación no lineal.

**Parámetros típicos (Keras / TensorFlow).**

- **`filters`:** número de mapas de salida (profundidad de la salida).
- **`kernel_size`:** tamaño espacial del filtro (ej. $3 \times 3$).
- **`padding`:** `"valid"` (sin relleno) o `"same"` (relleno para conservar tamaño espacial aproximado).
- **`strides`:** desplazamiento del kernel (por defecto 1).

**Ejemplo teórico.** Un kernel $3 \times 3$ sobre un mapa de $32 \times 32$ píxeles recorre posiciones vecinas y produce un mapa de activaciones que resalta, por ejemplo, bordes verticales según los pesos aprendidos.

**En este proyecto (CNN desde cero).**

```python
Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
BatchNormalization(),
MaxPooling2D(pool_size=(2, 2)),
Dropout(0.25),
Conv2D(64, (3, 3), activation="relu", padding="same"),
# ... Conv2D(128, ...)
```

**Ubicación:** función `run_cnn_from_scratch_training` en `scripts/training_utils.py`.

**Justificación.** Tres bloques convolucionales creciendo en profundidad (32 → 64 → 128) es un patrón clásico: primeras capas capturan detalles de bajo nivel; capas posteriores combinan información más abstracta. **`MaxPooling2D`** reduce dimensionalidad espacial y aporta cierta invariancia local a pequeñas traslaciones.

---

## Arquitecturas CNN para visión por computador

**Contexto teórico.** Históricamente aparecieron arquitecturas como **LeNet** (digitos), **AlexNet**, **VGG** (bloques repetidos), **ResNet** (conexiones residuales), **Inception**, y familias **EfficientNet** que equilibran profundidad, ancho y resolución para mejor eficiencia.

**Dos enfoques en este proyecto.**

| Enfoque | Descripción | Caminos en el repo |
|--------|-------------|-------------------|
| **CNN propia** | Pocos bloques conv + densas; entrenada desde cero sobre el dataset del curso. | 1 (my_images), 3 (FER/AffectNet) |
| **Backbone preentrenado** | **EfficientNetB0** entrenado en **ImageNet**; se sustituye solo la “cabeza” de clasificación por 4 clases. | 2 (my_images), 4 (FER/AffectNet) |

**En este proyecto — CNN desde cero.** Arquitectura secuencial explícita en `run_cnn_from_scratch_training`: convoluciones, pooling, aplanado `Flatten()`, capas `Dense` con regularización.

**En este proyecto — EfficientNet.**

```python
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
base_model.trainable = False  # fase 1: solo cabeza
# ... Sequential con Rescaling, Lambda(preprocess), base_model, GlobalAveragePooling2D, Dense, ...
```

Luego **fine-tuning:** se descongelan parcialmente las últimas capas de `base_model` y se reentrena con learning rate más bajo.

**Ubicación:** `run_transfer_efficientnet_training` en `scripts/training_utils.py`.

**Justificación.** EfficientNet es un estándar moderno para transfer en imágenes; ImageNet aporta filtros útiles para texturas y formas que luego se adaptan a rostros y emociones.

---

## Aumento de datos (data augmentation)

**Definición.** El **aumento de datos** consiste en aplicar **transformaciones aleatorias** a las imágenes de entrenamiento (rotaciones, traslaciones, cambios de brillo, etc.) para **aumentar la diversidad** efectiva del conjunto sin recoger nuevas fotos. Reduce **sobreajuste** y mejora la generalización si las transformaciones son plausibles para el dominio.

**Ejemplo teórico.** Una cara ligeramente girada o más oscura sigue siendo la misma emoción; el modelo debe ser robusto a esas variaciones.

**En este proyecto.** **Solo el conjunto de entrenamiento** usa `ImageDataGenerator` con augmentation; validación y test usan solo `rescale=1/255` (y conversión BGR→RGB si `DATA_SOURCE == "my_images"`).

```python
def get_train_datagen():
    return ImageDataGenerator(
        rescale=1.0 / 255,
        preprocessing_function=_preprocess_fn(),
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        zoom_range=0.3,
        shear_range=0.3,
        fill_mode="nearest",
    )
```

**Nota de diseño:** `vertical_flip=False` (comentado en código): en rostros, voltear verticalmente no es físicamente equivalente a una pose natural.

**Ubicación:** `scripts/data_preprocessing.py` — funciones `get_train_datagen`, `get_val_datagen`, `get_test_datagen` y generadores `flow_from_directory` con `target_size=(224, 224)`, `class_mode="categorical"`, `color_mode="rgb"`.

**Justificación.** Mismo preprocesado para todos los caminos que importan `get_train_generator` desde `training_utils.py`, evitando duplicar lógica y garantizando comparaciones justas entre experimentos.

---

## Aprendizaje por transferencia (transfer learning)

**Definición.** El **transfer learning** reutiliza un modelo (o parte de él) entrenado en un **dominio fuente** grande (p. ej. ImageNet) y lo **adapta** a un **dominio objetivo** más pequeño (p. ej. cuatro emociones en este curso). Se ahorra tiempo de entrenamiento y a menudo se mejora la precisión con pocos datos.

**Esquema típico.**

1. Cargar backbone con pesos preentrenados; **congelar** sus capas.
2. Entrenar solo las capas nuevas (cabeza de clasificación).
3. **Descongelar** parcialmente el backbone y entrenar con **learning rate bajo** (*fine-tuning*).

**En este proyecto (EfficientNet).**

```python
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
base_model.trainable = False
# ... entrenar cabeza con Adam(1e-4) ...
base_model.trainable = True
fine_tune_at = int(len(base_model.layers) * 0.8)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), ...)
# ... segundo fit ...
```

**Caminos 5 y 6** cargan un `.keras` previo (`load_model`), eliminan la última capa (`model.pop()`), congelan el resto y añaden una nueva `Dense(..., softmax)` para el nuevo número de clases — otro patrón de transfer desde **tu propio modelo** entrenado antes.

**Ubicación:** `run_transfer_efficientnet_training` y `run_transfer_from_existing_model_training` en `scripts/training_utils.py`.

**Justificación pedagógica.** Compara “desde cero” vs “ImageNet” vs “transfer desde modelo del curso”, alineado con los seis caminos del README.

---

## Frameworks de aprendizaje profundo

### TensorFlow

**Qué es.** **TensorFlow** es una biblioteca de código abierto de Google para cómputo numérico y aprendizaje automático. Proporciona tensores, operaciones, **diferenciación automática**, ejecución en **CPU y GPU**, y la API **tf.keras** para construir y entrenar redes.

**Instalación (típica).**

```bash
pip install tensorflow
```

En este proyecto las versiones concretas están acotadas en `requirements.txt` del repositorio.

**Uso en el proyecto.** Importación de `tensorflow as tf`, capas `tf.keras.layers`, optimizadores `tf.keras.optimizers`, `model.compile`, `model.fit`, `model.evaluate`, `model.predict`. Los grafos de computación se construyen al definir el modelo y se ejecutan en `fit`/`predict`.

**Grafos de computación.** El modelo es un grafo dirigido: nodos = operaciones (convolución, suma, activación); aristas = tensores. TensorFlow puede **fusionar** operaciones y lanzar kernels en GPU. En TF2 predomina la ejecución *eager* interactiva, pero la compilación interna (p. ej. `@tf.function`) sigue aprovechando grafos para rendimiento.

**Archivos relevantes.** `scripts/training_utils.py`, `scripts/data_preprocessing.py`, `scripts/realtime_emotion_recognition.py`, `scripts/run_model_path.py`.

---

### Keras

**Qué es.** **Keras** es una API de alto nivel para definir y entrenar redes: modelos `Sequential` o funcionales, capas reutilizables, callbacks (`EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`), generadores de imágenes.

**Relación con TensorFlow.** En este proyecto se usa **Keras integrado en TensorFlow** (`tf.keras`), no el paquete independiente `keras` en solitario. Evita conflictos de versiones entre `keras` y `tensorflow`.

**Instalación.** Viene con `pip install tensorflow`. No hace falta instalar `keras` por separado para este repositorio.

**Uso en el proyecto.** Definición de arquitectura, `compile`, `fit`, callbacks, y `ImageDataGenerator` en preprocesado.

---

### PyTorch

**Qué es.** **PyTorch** (Meta) es otro framework muy usado en investigación y producción. Modelos se suelen definir como clases `torch.nn.Module`; el bucle de entrenamiento es explícito (forward, loss, `backward()`, `optimizer.step()`).

**Instalación.**

```bash
pip install torch torchvision
```

**Uso típico.** Investigación con flexibilidad fina del grafo (*dynamic graph*). **Este proyecto no usa PyTorch**; se menciona como alternativa industrial y académica. Los conceptos (pérdida, gradiente, convolución) son los mismos que en TensorFlow.

---

### JAX

**Qué es.** **JAX** combina NumPy-like API con compilación **XLA** y transformaciones (gradientes, vectorización). Muy usado en investigación avanzada.

**Instalación.** `pip install jax jaxlib` (instrucciones oficiales según sistema y GPU).

**Este proyecto.** No utiliza JAX. Referencia para contextualizar el ecosistema.

---

### Resumen comparativo (para exposición)

| Framework | Rol típico | ¿Usado aquí? |
|-----------|------------|--------------|
| TensorFlow + tf.keras | Producción, curso, GPU en Colab | **Sí** |
| PyTorch | Investigación, muchos papers | No |
| JAX | Investigación, alto rendimiento | No |

---

## Aspectos prácticos del entrenamiento en redes profundas

### Unidades de activación

**ReLU:** $\text{ReLU}(x) = \max(0, x)$. Mitiga gradientes que se anulan en capas profundas (comparado con sigmoid/tanh en capas ocultas).

**Softmax:** $\text{softmax}(z)_i = e^{z_i} / \sum_j e^{z_j}$. Salida probabilística multiclase.

**En el proyecto.** `activation="relu"` en convoluciones y densas ocultas; `activation="softmax"` en la última `Dense`. Ver `scripts/training_utils.py`.

---

### Inicialización de parámetros

**Idea.** Pesos iniciales aleatorios con varianza acotada (Glorot/He) evitan que las activaciones exploten o desaparezcan.

**En el proyecto.** Inicialización por defecto de Keras en `Conv2D` y `Dense` sin `kernel_initializer` explícito.

---

### Normalización por lotes (batch normalization)

**Idea.** Estandarizar activaciones dentro de cada minibatch durante el entrenamiento; parámetros aprendibles de escala y desplazamiento.

**En el proyecto.** `BatchNormalization()` tras bloques conv y densos en CNN desde cero y en la cabeza de EfficientNet. `scripts/training_utils.py`.

---

### Optimización avanzada

**Adam** + **ReduceLROnPlateau** ya descritos arriba. Complemento: **`ModelCheckpoint`** guarda el mejor modelo según `val_loss` o `val_accuracy`; **`EarlyStopping`** detiene si no hay mejora en validación.

---

### Regularización

**Dropout:** durante el entrenamiento, se anula aleatoriamente una fracción de unidades (p. ej. 0.25, 0.5) para reducir co-adaptación.

**Early stopping:** criterio sobre `val_loss`, `patience=10`, `restore_best_weights=True`.

**En el proyecto.** `Dropout` y callbacks en `scripts/training_utils.py`.

---

## Tabla de correspondencia teoría–código

| Concepto | Archivo(s) principal(es) | Indicación |
|----------|--------------------------|------------|
| Pérdida categórica | `training_utils.py` | `loss="categorical_crossentropy"` |
| Adam / ReduceLROnPlateau | `training_utils.py` | `compile`, `callbacks` |
| Backprop / entrenamiento | `training_utils.py` | `model.fit(...)` |
| TensorFlow / Keras | Todo el pipeline de scripts | imports `tf.keras` |
| Conv2D / MaxPooling | `training_utils.py` | `run_cnn_from_scratch_training` |
| EfficientNet / transfer | `training_utils.py` | `run_transfer_efficientnet_training`, `run_transfer_from_existing_model_training` |
| Data augmentation | `data_preprocessing.py` | `get_train_datagen`, `ImageDataGenerator` |
| Generadores train/val/test | `data_preprocessing.py` | `get_*_generator` |
| Inferencia | `realtime_emotion_recognition.py` | `load_model`, `predict` |

---

## Síntesis para exposición oral

Este trabajo implementa **clasificación de emociones en imágenes** con redes convolucionales. Se minimiza la **entropía cruzada** entre la salida **softmax** y las etiquetas one-hot; **TensorFlow/Keras** calcula **gradientes por retropropagación** y actualiza pesos con **Adam** y políticas de **learning rate** y **early stopping**. La **CNN desde cero** apila **convoluciones, pooling, BatchNorm y Dropout**; los caminos con **EfficientNet** usan **transfer learning** desde **ImageNet** y **fine-tuning**. El **aumento de datos** en entrenamiento está centralizado en **`data_preprocessing.py`**, lo que hace comparable y justificable todo el flujo experimental descrito en el README del repositorio.

---

*Documento generado como material de teoría alineado al código del proyecto ReconocimientoEmociones-CNN. Para el flujo operativo (venv, Colab, scripts), ver `README.md` y `NOTAS_TECNICAS.txt`.*
