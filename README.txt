Librerias necesarias: crea un scrip, para pobar si tienes instaladas cada una de ellas

Codigo: cipia y pega

import tensorflow as tf  # pip install tensorflow
import keras  # pip install keras
import numpy as np  # pip install numpy
import cv2  # pip install opencv-python
import sklearn  # pip install scikit-learn
import shutil  #pip install pytest-shutil

print(tf.__version__)
print(keras.__version__)
print(np.__version__)
print(cv2.__version__)
print(sklearn.__version__)

-----------------------------------------------------
Distribución de Datos

Train (Entrenamiento):
Propósito: Utilizado directamente por el modelo para aprender las características de las clases.
Cantidad: Generalmente, el 70-80% del total de tus datos.
Ejemplo: Si tienes 800 imágenes en total (200 por cada emoción), puedes usar 140-160 imágenes por emoción para entrenamiento.

Validation (Validación):
Propósito: Utilizado para ajustar hiperparámetros y monitorear el desempeño del modelo durante el entrenamiento, sin sesgar la evaluación final.
Cantidad: Un 10-15% de tus datos.
Ejemplo: Puedes usar 20-30 imágenes por emoción para validación.

Test (Prueba):
Propósito: Evaluación final del modelo después del entrenamiento para obtener una métrica de rendimiento objetiva.
Cantidad: Otro 10-15% de tus datos.
Ejemplo: Reserva 20-30 imágenes por emoción para prueba.

-------------------------------------------------------------

Descripcion de Scrips

data_collection> Este script utiliza OpenCV para capturar imágenes de rostros en tiempo real a través de la webcam de una computadora.
Los rostros capturados se guardan como imágenes de 48x48 píxeles en escala de grises. Cada imagen se guarda en una carpeta correspondiente a una emoción predefinida. 
El programa permite al usuario empezar y terminar la captura de imágenes mediante la pulsación de teclas.

data_split: Este script se utiliza para organizar un conjunto de imágenes en subconjuntos de entrenamiento, validación y prueba, lo cual es un paso fundamental en la 
preparación de los datos para el entrenamiento de modelos de inteligencia artificial y aprendizaje automático. Las imágenes se dividen en tres subconjuntos y se organizan 
en carpetas de acuerdo con la clase o categoría de emoción y el subconjunto al que pertenecen (entrenamiento, validación o prueba). Este tipo de organización es una 
convención común en los flujos de trabajo de visión por computadora y permite manipular y acceder fácilmente a los datos durante el entrenamiento y la validación de los modelos.

data_processing: Este script tiene como objetivo preprocesar y cargar las imágenes que se utilizarán para entrenar un modelo de aprendizaje profundo. 
En esta tarea se incluye la normalización, augmentación de los datos y la separación de las imágenes en conjuntos de entrenamiento y validación. 
Al terminar, se dispondrá de generadores de datos listos para ser usados en el entrenamiento y validación del modelo. 
La augmentación de los datos es especialmente útil para prevenir el sobreajuste, incrementar la calidad de datos y hacer que el modelo sea más robusto a variaciones en los datos.

model_training: Este script es responsable del entrenamiento de un modelo para la clasificación de emociones, utilizando una CNN pre-entrenada llamada VGG16. Luego de construir el modelo, 
se entrena con las imágenes que se han preprocesado y dividido en conjuntos de entrenamiento y validación. Luego se evalúa su rendimiento con un conjunto de prueba. Los callbacks 
implementados en este script permiten guardar el mejor modelo durante el entrenamiento, detener el entrenamiento si el rendimiento del modelo no mejora tras varios epochs (Early Stopping)
y ajustar la tasa de aprendizaje del optimizador durante el entrenamiento (ReduceLROnPlateau).

realtime_emotion_recognition:  es un script de reconocimiento de emociones en tiempo real que captura video de una cámara, detecta rostros en cada frame, y usa un modelo de aprendizaje 
profundo para detectar la emoción en cada rostro detectado. Muestra las emociones detectadas como etiquetas de texto en el video.

custom_cnn: es basicamente para generar tu propio modelo CNN, solo es como referencia para ver como es que se generan.