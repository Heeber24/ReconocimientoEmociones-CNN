================================================================================
  RECONOCIMIENTO DE EMOCIONES CON REDES NEURONALES CONVOLUCIONALES (CNN)
  Evidencia integradora - Sistemas Cognitivos Artificiales
================================================================================

  Este proyecto reproduce un flujo real: captura o datos, preprocesado,
  entrenamiento (Transfer Learning o red desde cero) y reconocimiento en
  tiempo real. Con la documentación basta seguir pasos y ajustar
  configuraciones según lo que quieras experimentar.


INICIO RÁPIDO
-------------
  1. pip install -r requirements.txt
  2. Genera imágenes (data_collection.py) o prepara un dataset con
     carpetas por emoción en data/images/.
  3. data_split.py → data_preprocessing.py → model_training.py O custom_cnn.py
  4. realtime_emotion_recognition.py para ver el modelo en la cámara.

  Todo se ejecuta desde la raíz del proyecto. Rutas y opciones en
  scripts/config.py; detalles más abajo.


LOS 6 CAMINOS DEL FLUJO
------------------------
  El código está pensado para que puedas seguir cualquiera de estos seis
  caminos; lo que cambia es la fuente de datos, el script de entrenamiento
  y qué opciones tocas en config y en realtime. Más abajo se indica qué
  modificar en cada caso para no tener problemas. Todos pasan por
  data_split y data_preprocessing; los caminos 5 y 6 usan además
  transfer_from_custom.py y llevan dentro el 1 o el 3 como primera fase.

  1. Tu dataset (data_collection) → data_split → data_preprocessing
     → Red CNN propia (custom_cnn.py).

  2. Tu dataset (data_collection) → data_split → data_preprocessing
     → Red con Transfer, p. ej. EfficientNetB0 (model_training.py).

  3. Repositorio de imágenes → data_split → data_preprocessing
     → Red CNN propia (custom_cnn.py).

  4. Repositorio de imágenes → data_split → data_preprocessing
     → Red con Transfer, p. ej. EfficientNetB0 (model_training.py).

  5. (Implica primero el camino 1.) Tu dataset → data_split → data_preprocessing
     → custom_cnn (red propia con tus imágenes). Después: repositorio →
     data_split → data_preprocessing → transfer_from_custom.py (cargas ese
     modelo como base y entrenas con el repo; resultado en models/).

  6. (Implica primero el camino 3.) Repositorio → data_split → data_preprocessing
     → custom_cnn (red propia con el repo). Después: tu dataset → data_split →
     data_preprocessing → transfer_from_custom.py (cargas ese modelo como base
     y entrenas con tus imágenes; resultado en models/).

  En 5 y 6: en transfer_from_custom.py pones MODEL_AS_BASE en el .keras que
  corresponda y usas los datos del otro origen (repo o tuyos) para la segunda fase.

  En realtime: si terminaste con model_training → TRANSFER_MODEL_NAME = "EfficientNetB0" (u otra base).
  Si terminaste con custom_cnn → use_custom_cnn = True.
  Si terminaste con transfer_from_custom → use_from_custom = True.


QUÉ MODIFICAR PARA CADA CAMINO (y así no tener problemas)
----------------------------------------------------------
  Para que al correr el camino que elijas no falle ni cargue datos equivocados,
  revisa lo siguiente. Todo lo que no se menciona déjalo por defecto.

  Camino 1 – Tu dataset → custom_cnn
    • Datos: captura con data_collection.py (las imágenes quedan en
      data/images/data_collection/<emotion>/). No cambies config.py.
    • Orden: data_collection → data_split → data_preprocessing → custom_cnn.py.
    • En realtime_emotion_recognition.py: use_custom_cnn = True,
      use_from_custom = False.

  Camino 2 – Tu dataset → model_training (Transfer)
    • Datos: igual que camino 1 (data_collection, data_split, data_preprocessing).
    • En model_training.py: BASE_MODEL_NAME = "EfficientNetB0" (o "VGG16", etc.),
      según la base que quieras.
    • Orden: data_collection → data_split → data_preprocessing → model_training.py.
    • En realtime: use_custom_cnn = False, use_from_custom = False,
      TRANSFER_MODEL_NAME = "EfficientNetB0" (o la misma base que usaste).

  Camino 3 – Repositorio → custom_cnn
    • Datos: pon el dataset del repo en data/images/data_collection con
      subcarpetas por emoción (angry, happy, neutral, surprise), o prepara
      el repo en una carpeta que tenga train/validation/test y en config.py
      pon DATA_ROOT = Path("ruta/a/esa/carpeta").
    • Si usas data_collection: copia las imágenes del repo ahí, luego
      data_split → data_preprocessing → custom_cnn.py. No toques DATA_ROOT.
    • En realtime: use_custom_cnn = True, use_from_custom = False.

  Camino 4 – Repositorio → model_training (Transfer)
    • Datos: igual que camino 3 (repo en data_collection y data_split, o
      DATA_ROOT apuntando al prepared del repo).
    • En model_training.py: BASE_MODEL_NAME = la base que quieras.
    • Orden: (preparar repo) → data_split si hace falta → data_preprocessing
      → model_training.py.
    • En realtime: use_custom_cnn = False, use_from_custom = False,
      TRANSFER_MODEL_NAME = la misma base que BASE_MODEL_NAME.

  Camino 5 – Primero camino 1, luego transfer con repo
    • Fase 1: haz el camino 1 completo (tus imágenes → custom_cnn).
      Queda guardado models/emotion_recognition_Personal.keras.
    • Fase 2: deja que los datos de entrenamiento sean del REPO. Para eso:
      pon el repo en data/images/data_collection (carpetas por emoción) y
      ejecuta data_split de nuevo (prepared_data pasará a ser el split del
      repo), o pon el repo ya partido en train/val/test y en config.py
      pon DATA_ROOT = Path("ruta/al/repo/prepared").
      Luego ejecuta data_preprocessing y transfer_from_custom.py.
    • En transfer_from_custom.py: MODEL_AS_BASE = MODEL_CUSTOM (por defecto),
      así se carga el Personal.keras que entrenaste con tus imágenes.
    • En realtime (para el modelo final de la fase 2): use_from_custom = True.

  Camino 6 – Primero camino 3, luego transfer con tus imágenes
    • Fase 1: haz el camino 3 completo (repo → custom_cnn). Queda guardado
      models/emotion_recognition_Personal.keras (entrenado con repo).
    • Fase 2: deja que los datos de entrenamiento sean TU DATASET. Para eso:
      pon de nuevo tus imágenes en data/images/data_collection (o en una
      carpeta con train/val/test) y en config.py pon DATA_ROOT = PREPARED_DATA
      (o la ruta a tu prepared). Ejecuta data_split si cambiaste data_collection,
      luego data_preprocessing y transfer_from_custom.py.
    • En transfer_from_custom.py: MODEL_AS_BASE = MODEL_CUSTOM (por defecto),
      así se carga el Personal.keras entrenado con el repo.
    • En realtime (modelo final): use_from_custom = True.

  Resumen de archivos que tocar según el camino:
  • config.py: DATA_ROOT solo si usas repo (caminos 3–6) en alguna fase.
  • model_training.py: BASE_MODEL_NAME para caminos 2 y 4.
  • transfer_from_custom.py: MODEL_AS_BASE para 5 y 6 (por defecto MODEL_CUSTOM).
  • realtime_emotion_recognition.py: use_custom_cnn / use_from_custom /
    TRANSFER_MODEL_NAME según con qué modelo quieras probar en vivo.


NOTA SOBRE IDIOMA
-----------------
  Código y carpetas están en inglés; este README en español. data/images
  es donde van tus capturas o el dataset. Si tenías Tus_Imagenes, renómbrala a "images".


CONTEXTO DE LA ASIGNATURA
-------------------------
Este proyecto pone en práctica contenidos de la asignatura Sistemas Cognitivos
Artificiales, en particular:

  • Tema 2 - Entrenamiento de redes neuronales: función de coste (categorical
    cross-entropy), optimización (Adam / gradient descent), backpropagation.
  • Tema 3 - Frameworks: uso de TensorFlow y Keras para definir y entrenar
    modelos.
  • Tema 4 - Aspectos prácticos: unidades de activación (ReLU, softmax),
    inicialización, Batch Normalization, regularización (Dropout), optimización
    avanzada (ReduceLROnPlateau, Early Stopping).
  • Tema 5 - CNN: capas convolucionales, arquitecturas para visión, data
    augmentation y Transfer Learning (modelo preentrenado EfficientNetB0).
  • Tema 9 (opcional) - Uso de GPU para acelerar el entrenamiento.

El ejercicio es un caso real de visión por computador: clasificación de
emociones en rostros usando una red neuronal convolucional (ya sea construida
desde cero o por Transfer Learning).


OBJETIVO PEDAGÓGICO
-------------------
  1. Entrenar con imágenes propias (webcam) y con un dataset de repositorio;
     comparar la precisión en test y reflexionar sobre la importancia de
     la calidad y cantidad de los datos.
  2. Entrenar un modelo desde cero (custom_cnn.py) y otro con Transfer
     Learning (model_training.py); comparar precisión, tiempo y ventajas
     de cada enfoque.
  3. Sintetizar conclusiones sobre datos, entrenamiento y tipo de modelo
     (desde cero vs transfer learning).


REQUISITOS E INSTALACIÓN
-------------------------
  • Python 3.8 o superior.
  • Dependencias: TensorFlow, NumPy, OpenCV, scikit-learn.

  Recomendación: usar un entorno virtual (venv) para que las bibliotecas
  del proyecto no se mezclen con las del sistema u otros proyectos. Así
  cada estudiante tiene "su propio entorno" para este ejercicio.

  Opción A – Desde la raíz del proyecto (recomendado):
    1. Crear y activar el entorno:
         Windows:  python -m venv venv   luego   venv\Scripts\activate
         Linux/Mac: python3 -m venv venv  luego   source venv/bin/activate
    2. Con el entorno activado, instalar:
         pip install -r requirements.txt
    Decimos "desde la raíz" para que la ruta requirements.txt sea la actual;
    con el venv activo, pip instala las librerías dentro de ese entorno.

 
  Comprobar que todo está instalado:

    python scripts/bibliotecas_versiones.py


ESTRUCTURA DEL PROYECTO
-------------------------
  ReconocimientoEmociones-CNN/
  ├── scripts/
  │   ├── config.py                    # Rutas y parámetros comunes
  │   ├── bibliotecas_versiones.py     # Comprueba dependencias
  │   ├── data_collection.py           # Captura de rostros por webcam
  │   ├── data_split.py                # División train/validation/test
  │   ├── data_preprocessing.py        # Preprocesado (normalización + augmentation) + validación
  │   ├── model_training.py            # Transfer con base ImageNet (VGG16, EfficientNet, etc.)
  │   ├── custom_cnn.py                # CNN desde cero
  │   ├── transfer_from_custom.py      # Transfer usando tu modelo guardado como base
  │   └── realtime_emotion_recognition.py  # Carga un modelo de models/ y reconoce en vivo
  ├── data/
  │   └── images/                      # Aquí van tus capturas o el dataset
  │       ├── data_collection/         # Imágenes crudas por emoción (webcam)
  │       └── prepared_data/           # train / validation / test
  ├── models/                          # Modelos guardados (.keras)
  ├── requirements.txt
  ├── README.txt (este archivo)
  └── NOTAS_TECNICAS.txt               # Épocas, capas, pesos, diferencias
                                       # Transfer vs CNN desde cero (para clase)


DISTRIBUCIÓN DE DATOS: TRAIN / VALIDATION / TEST
-------------------------------------------------
  • Train (entrenamiento): ~70% de los datos. El modelo aprende con ellos.
  • Validation (validación): ~15%. Para ajustar hiperparámetros y monitorear
    el entrenamiento sin evaluar “en exceso” sobre los mismos datos.
  • Test (prueba): ~15%. Evaluación final para una métrica objetiva.

  Ejemplo: 800 imágenes en total (200 por emoción) → ~560 train, ~120
  validación, ~120 test por clase. Las proporciones se configuran en
  data_split.py (VALIDATION_TEST_SPLIT_SIZE, TEST_SPLIT_FROM_REMAINDER).

  ¿Por qué tres conjuntos y no solo train y test? Si solo usas train y
  test, cualquier decisión tomada mirando el test (épocas, learning
  rate, etc.) hace que el test deje de ser una estimación imparcial
  (fuga de información) e infla la precisión reportada. El estándar en
  literatura (Goodfellow et al. "Deep Learning", cursos de ML) es:
  train (aprender), validation (tuning/parada), test (métrica final).


TAMAÑO DE IMAGEN Y COLOR: QUÉ DICE LA LITERATURA
-------------------------------------------------
  Tamaño: 224×224 es estándar para modelos preentrenados (VGG, ResNet,
  EfficientNet) y permite Transfer Learning sin cambiar la arquitectura.
  En emociones también se usan 48×48 o 64×64 (ej. FER2013): menos
  píxeles, menos parámetros y entrenamiento más rápido, pero menos
  detalle. Este proyecto usa 224×224 para ser coherente con EfficientNetB0.

  Color vs escala de grises: Escala de grises (1 canal) es común en
  datasets clásicos (FER2013, CK+); reduce dimensiones. Color (RGB, 3
  canales) lo esperan los modelos ImageNet y puede aportar iluminación
  y tono de piel. Para Transfer Learning con ImageNet hay que usar RGB
  (224×224×3). Para una CNN pequeña desde cero se puede usar gris para
  ahorrar parámetros. Este proyecto usa RGB y 224×224 en todo el pipeline.


PASOS EN ORDEN (DETALLE)
-------------------------
  Siempre desde la raíz del proyecto. El orden es fijo hasta el paso 3;
  en el 4 eliges Transfer o red desde cero.

  Paso | Script                     | Qué hace
  -----+----------------------------+------------------------------------------
  1    | data_collection.py         | Captura rostros por webcam (o usas repo)
  2    | data_split.py              | Divide en train / validation / test
  3    | data_preprocessing.py      | Preprocesado + validación (obligatorio)
  4    | model_training.py O        | Entrenar: Transfer Learning o CNN propia
       | custom_cnn.py              |
  5    | realtime_emotion_recognition.py | Probar el modelo en vivo

  Si usas dataset de repo: ponlo en data/images con train/validation/test
  (o cambia DATA_ROOT en config.py) y omite o adapta 1 y 2. El preprocesado
  (normalización y data augmentation) está solo en data_preprocessing.py;
  model_training y custom_cnn solo importan esos generadores y entrenan.


MODELOS DE TRANSFER LEARNING DISPONIBLES
----------------------------------------
  Por defecto se usa EfficientNetB0. Puedes probar otros y comparar precisión.
  En model_training.py cambia la variable BASE_MODEL_NAME al inicio del script.
  En realtime_emotion_recognition.py cambia TRANSFER_MODEL_NAME para cargar
  el mismo modelo que entrenaste.

  Modelos disponibles (todos 224×224, ImageNet):
  • EfficientNetB0  (por defecto; buen equilibrio precisión/tiempo)
  • VGG16          (clásico, más lento, muchos parámetros)
  • ResNet50       (muy usado, buena precisión)
  • MobileNetV2    (más ligero y rápido, menos parámetros)
  • DenseNet121    (buena precisión, arquitectura densa)

  Prueba distintos y anota la precisión en test; así ves si en tu dataset
  alguno rinde mejor.


MISMO ROSTRO (TUS IMÁGENES) Y OTRAS PERSONAS
---------------------------------------------
  Si entrenas solo con tu propio rostro (las capturas son solo tuyas):
  • Para TI: el modelo "conoce" tu firma facial; suele funcionar bien
    para reconocerte a ti mismo en tiempo real. Tiene cierto sesgo hacia
    tu fisonomía, pero es esperado.
  • Para OTRAS PERSONAS: el modelo no ha visto otros rostros; puede
    funcionar peor con otras caras (menor precisión o más confusión).
    Para que generalice bien a cualquiera, haría falta entrenar con
    imágenes de muchas personas (p. ej. dataset público con múltiples
    identidades). Para este proyecto está bien que sea solo tu rostro;
    si quieres probar con otros, pídeles que pasen por la cámara y
    observa si las predicciones siguen siendo razonables.


¿CUÁNTAS IMÁGENES? (LÍMITE POR EMOCIÓN)
----------------------------------------
  Por defecto el script de captura guarda hasta 50 imágenes por emoción
  (variable MAX_IMAGES en data_collection.py). Con 4 emociones son 200
  imágenes en total.

  • 200 imágenes (50 por emoción) suelen ser suficientes para empezar
    y para que el modelo te reconozca a ti en tiempo real, sobre todo
    con Transfer Learning. Puedes subir el límite (p. ej. 80 o 100 por
    emoción) en data_collection.py si quieres más robustez.
  • Para generalizar bien a otras personas, idealmente más imágenes y
    de varias identidades (dataset de repositorio o más gente capturando).


QUÉ HACE CADA SCRIPT (RESUMEN)
------------------------------
  data_collection.py   Webcam: eliges emoción (angry, happy, neutral, surprise),
                       tecla 'C' inicia/pausa, ESC sale. Guarda 224x224 en
                       data/images/data_collection/<emotion>/.

  data_split.py        Reparte esas imágenes en train/validation/test
                       (data/images/prepared_data/). Proporciones en el script.

  data_preprocessing.py  Define normalización y data augmentation; crea los
                       generadores que usan los scripts de entrenamiento. Al
                       ejecutarlo valida carpetas y muestra conteo. Opción
                       --data-root <ruta> para validar otro dataset (repo).

  model_training.py    Transfer Learning: base preentrenada + cabeza, entrena,
                       evalúa en test. Guarda en models/emotion_recognition_<Base>_model.keras.
                       Cambia BASE_MODEL_NAME al inicio para otro modelo.

  custom_cnn.py        CNN desde cero: entrena y guarda en
                       models/emotion_recognition_Personal.keras.

  transfer_from_custom.py   Carga un modelo que ya tengas en models/ (p. ej.
                       custom_cnn), lo usa como base, entrena nueva cabeza y
                       guarda emotion_recognition_from_custom.keras en models/.

  realtime_emotion_recognition.py  Carga el modelo, abre cámara y muestra
                       la emoción. Ajusta TRANSFER_MODEL_NAME, use_custom_cnn
                       o use_from_custom según el .keras que quieras usar.

  config.py            Rutas y parámetros comunes. DATA_ROOT = raíz de
                       train/validation/test; cámbialo para usar un repo.


COMPARACIONES EVALUABLES (datos y tipo de modelo)
-------------------------------------------------
  A) Comparación por FUENTE DE DATOS
  1. Entrenamiento con tus imágenes: data_collection.py → data_split.py
     → model_training.py. Anota precisión en test.
  2. Entrenamiento con dataset de repositorio: el que indique el profesor,
     misma estructura de carpetas por emoción → data_split.py → entrena
     de nuevo. Anota precisión en test.
  3. Compara ambas precisiones y redacta por qué pueden diferir y la
     importancia de un buen conjunto de datos.

  B) Comparación por TIPO DE MODELO (desde cero vs Transfer Learning)
     1. Con el mismo dataset (p. ej. repositorio), entrena
        model_training.py (Transfer Learning) y custom_cnn.py (CNN desde
        cero). Anota precisión en test y tiempo (o épocas) de cada uno.
     2. Redacta ventajas y desventajas: desde cero (control, más datos y
        tiempo, didáctico) vs Transfer Learning (mejor precisión con
        menos datos/tiempo, conocimiento preentrenado).
  C) Síntesis: conclusión breve sobre qué influye más (datos vs tipo de
     modelo) y cuándo elegirías cada enfoque en un proyecto real.


DESCRIPCIÓN DETALLADA DE SCRIPTS
---------------------------------
  data_collection.py
    Usa OpenCV para capturar rostros en tiempo real con la webcam. Los
    rostros se redimensionan a 224x224 píxeles y se guardan en carpetas
    por emoción. Menú para elegir emoción y cámara; tecla 'C' para
    iniciar/pausar, ESC para salir.

  data_split.py
    Toma las imágenes de data_collection (o de la carpeta que uses como
    origen) y las reparte en train, validation y test con proporciones
    configurables, manteniendo la estructura por clase (emoción). Usa
    sklearn.train_test_split con semilla fija para reproducibilidad.

  data_preprocessing.py
    Único lugar donde se define el preprocesado: normalización 1/255,
    data augmentation (rotación, shift, flip, brillo, zoom, shear) y
    creación de generadores. Exporta get_train_generator, get_validation
    _generator, get_test_generator para que model_training y custom_cnn
    los importen. Al ejecutarlo como script valida train/ y validation/
    (por defecto config.DATA_ROOT) y muestra conteo por emoción. Opción
    --data-root <ruta> para validar datos externos (repo o otra carpeta).
    Para entrenar con datos de repo: en config.py cambia DATA_ROOT a la
    ruta que tenga train/, validation/, test/ con subcarpetas por emoción.

  model_training.py
    Solo Transfer Learning: construye modelo (base preentrenada + cabeza),
    entrena y evalúa en test. Usa generadores de data_preprocessing.
    Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau; fine-tuning opcional.

  realtime_emotion_recognition.py
    Reconocimiento en tiempo real: captura de cámara, detección de rostros,
    redimensionado a 224x224, normalización y predicción con el modelo
    cargado. Muestra la etiqueta de la emoción sobre cada rostro.
    Las etiquetas deben estar en orden alfabético (angry, happy, neutral,
    surprise) para coincidir con flow_from_directory.

  custom_cnn.py
    Solo red desde cero: construye la CNN y entrena. Usa generadores de
    data_preprocessing; no define preprocesado.


DÓNDE CONFIGURAR QUÉ
--------------------
  • Usar otro dataset (repo, otra carpeta)  → config.py: DATA_ROOT
  • Límite de imágenes por emoción (captura) → data_collection.py: MAX_IMAGES
  • Proporciones train/validation/test      → data_split.py: VALIDATION_TEST_SPLIT_SIZE, etc.
  • Modelo de Transfer (VGG16, ResNet50…)   → model_training.py: BASE_MODEL_NAME
  • Qué modelo cargar en tiempo real        → realtime: TRANSFER_MODEL_NAME, use_custom_cnn o use_from_custom
  • Épocas, learning rate, dropout          → model_training.py o custom_cnn.py (variables al inicio)


NOTAS TÉCNICAS
--------------
  • Rutas en scripts/config.py. DATA_ROOT = raíz de train/validation/test.
  • Preprocesado solo en data_preprocessing.py; entrenamiento solo en model_training o custom_cnn.
  • Imagen: 224x224; emociones: angry, happy, neutral, surprise (orden alfabético).
  • NOTAS_TECNICAS.txt: épocas, capas, pesos, tiempos, comparativas para clase.


SUGERENCIAS PARA GITHUB
------------------------
  Incluir README, requirements.txt, NOTAS_TECNICAS.txt y la estructura
  (scripts/, data/, models/). No subir datasets ni .keras pesados; los
  estudiantes clonan, pip install -r requirements.txt y siguen los pasos
  de este README. Varias vías (tus imágenes/repo, Transfer/custom_cnn)
  llevan al mismo destino: un modelo listo para realtime_emotion_recognition.

================================================================================
  Resumen: instala dependencias → datos (captura o repo) → data_split →
  data_preprocessing → entrena (model_training o custom_cnn) → realtime.
  Experimenta cambiando datos y tipo de modelo; la documentación te guía.
================================================================================
