[1mdiff --git a/NOTAS_TECNICAS.txt b/NOTAS_TECNICAS.txt[m
[1mindex 4812349..f45a2c9 100644[m
[1m--- a/NOTAS_TECNICAS.txt[m
[1m+++ b/NOTAS_TECNICAS.txt[m
[36m@@ -145,7 +145,7 @@[m [msentido.[m
   el modelo correcto.[m
 [m
 [m
[31m-6. QUÉ DESTACAR EN CLASE[m
[32m+[m[32m6. COSAS A DESTACAR[m[41m [m
 ------------------------[m
   • Épocas: son un tope máximo; el entrenamiento puede parar antes por[m
     early stopping. Se eligen según datos y complejidad del modelo.[m
[1mdiff --git a/scripts/config.py b/scripts/config.py[m
[1mindex 6ce6e0e..53fd92b 100644[m
[1m--- a/scripts/config.py[m
[1m+++ b/scripts/config.py[m
[36m@@ -11,7 +11,7 @@[m [mPROJECT_ROOT = Path(__file__).resolve().parent.parent[m
 [m
 # Dataset: raw captures and train/validation/test splits (see README)[m
 # Si usas FER2013 en data/kaggle_fer/, pon USE_KAGGLE_FER = True.[m
[31m-USE_KAGGLE_FER = True  # True = leer desde data/kaggle_fer para data_split[m
[32m+[m[32mUSE_KAGGLE_FER = True  # True = leer desde data/..0.0......00000000000000000000000000000000000.................................................0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000kagg00000000000000000000000000000000000000000000000000000000000000000le_fer para data_split[m
 DATA_COLLECTION = (PROJECT_ROOT / "data" / "kaggle_fer") if USE_KAGGLE_FER else (PROJECT_ROOT / "data" / "images" / "data_collection")[m
 PREPARED_DATA = PROJECT_ROOT / "data" / "images" / "prepared_data"[m
 [m
[36m@@ -36,5 +36,11 @@[m [mEMOTION_LIST = ["angry", "happy", "neutral", "surprise"][m
 # Parámetros de imagen (deben ser coherentes en captura, preprocesado y modelo)[m
 FACE_SIZE = (224, 224)[m
 IMG_SHAPE = (224, 224, 3)[m
[31m-# True si las imágenes vienen de data_collection.py (OpenCV guarda BGR). False si usas FER2013 (kaggle_fer).[m
[31m-IMAGES_ARE_BGR = False  # False para FER2013 en kaggle_fer[m
[32m+[m[32m# True si las imágenes vienen de data_collection.py (OpenCV guarda BGR).[m
[32m+[m[32m# Para FER2013 en kaggle_fer, usamos ROI/inputs en escala de grises (o RGB replicado),[m
[32m+[m[32m# así que no invertimos canales.[m
[32m+[m[32mIMAGES_ARE_BGR = not USE_KAGGLE_FER[m
[32m+[m
[32m+[m[32m# En realtime, para FER2013 conviene usar ROI en escala de grises para que coincida mejor[m
[32m+[m[32m# con el entrenamiento. Para capturas webcam conviene usar ROI en color.[m
[32m+[m[32mUSE_GRAY_ROI = USE_KAGGLE_FER[m
[1mdiff --git a/scripts/realtime_emotion_recognition.py b/scripts/realtime_emotion_recognition.py[m
[1mindex 42fd09f..94cb6ab 100644[m
[1m--- a/scripts/realtime_emotion_recognition.py[m
[1m+++ b/scripts/realtime_emotion_recognition.py[m
[36m@@ -10,9 +10,8 @@[m [msys.path.insert(0, str(Path(__file__).resolve().parent))[m
 import cv2[m
 import numpy as np[m
 from keras.models import load_model  # type: ignore[m
[31m-from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore[m
 [m
[31m-from config import MODELS_DIR, MODEL_CUSTOM, MODEL_FROM_CUSTOM, EMOTION_LIST[m
[32m+[m[32mfrom config import MODELS_DIR, MODEL_CUSTOM, MODEL_FROM_CUSTOM, EMOTION_LIST, FACE_SIZE, IMAGES_ARE_BGR, USE_GRAY_ROI[m
 [m
 # Qué modelo cargar (elige uno):[m
 # - Transfer (ImageNet): use_custom_cnn = False, TRANSFER_MODEL_NAME = "EfficientNetB0", etc.[m
[36m@@ -60,23 +59,32 @@[m [mwhile cap.isOpened():  # Bucle principal para capturar y procesar fotogramas de[m
     faces = faceClassif.detectMultiScale(gray, 1.3, 5)  # Detecta rostros en el fotograma usando el clasificador Haar Cascade[m
 [m
     for (x, y, w, h) in faces:  # Itera sobre los rostros detectados[m
[31m-        roi_gray = gray[y:y + h, x:x + w]  # Extrae la región de interés (ROI) del rostro en escala de grises[m
[31m-        roi_gray = cv2.resize(roi_gray, (224, 224), interpolation=cv2.INTER_CUBIC)  # Redimensiona el ROI a 224x224 píxeles[m
[31m-        roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)  # Convierte el ROI a color BGR[m
[31m-        roi = roi_color.astype("float") / 255.0  # Normaliza los valores de los píxeles del ROI al rango [0, 1][m
[31m-        roi = img_to_array(roi)  # Convierte el ROI a un arreglo NumPy[m
[31m-        roi = np.expand_dims(roi, axis=0)  # Agrega una dimensión al arreglo para que coincida con la entrada del modelo[m
[31m-[m
[31m-        if roi.shape == (1, 224, 224, 3):  # Verifica si la forma del ROI es correcta (1, 224, 224, 3)[m
[31m-            try:[m
[31m-                prediction = model.predict(roi, verbose=0)  # Realiza la predicción de la emoción usando el modelo cargado (verbose=0 para evitar mensajes innecesarios)[m
[31m-                emotion_index = np.argmax(prediction)  # Obtiene el índice de la emoción con mayor probabilidad[m
[31m-                emotion = emotion_labels[emotion_index]  # Obtiene la etiqueta de la emoción correspondiente al índice[m
[31m-                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Muestra la emoción en el fotograma[m
[31m-            except Exception as e:  # Captura cualquier excepción que ocurra durante la predicción[m
[31m-                print(f"Error durante la predicción: {e}")  # Imprime un mensaje de error si ocurre una excepción[m
[31m-        else:  # Si la forma del ROI no es correcta[m
[31m-            print(f"Error: Forma de roi incorrecta: {roi.shape}")  # Imprime un mensaje de error indicando la forma incorrecta del ROI[m
[32m+[m[32m        if USE_GRAY_ROI:[m
[32m+[m[32m            # Recorte en gris para coincidir mejor con FER2013[m
[32m+[m[32m            roi = gray[y:y + h, x:x + w][m
[32m+[m[32m            roi = cv2.resize(roi, FACE_SIZE, interpolation=cv2.INTER_CUBIC)[m
[32m+[m[32m            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)  # 3 canales (BGR)[m
[32m+[m[32m        else:[m
[32m+[m[32m            # Recorte en color (webcam)[m
[32m+[m[32m            roi = frame[y:y + h, x:x + w][m
[32m+[m[32m            roi = cv2.resize(roi, FACE_SIZE, interpolation=cv2.INTER_CUBIC)  # 3 canales BGR[m
[32m+[m
[32m+[m[32m        roi = roi.astype("float32") / 255.0  # Normaliza al rango [0, 1][m
[32m+[m
[32m+[m[32m        # Si el entrenamiento convertía BGR->RGB, entonces lo hacemos aquí.[m
[32m+[m[32m        # (Para ROI gris, esta operación no cambia nada porque los canales son iguales.)[m
[32m+[m[32m        if IMAGES_ARE_BGR:[m
[32m+[m[32m            roi = roi[..., ::-1].copy()  # BGR -> RGB[m
[32m+[m
[32m+[m[32m        roi = np.expand_dims(roi, axis=0)  # (1, 224, 224, 3)[m
[32m+[m
[32m+[m[32m        try:[m
[32m+[m[32m            prediction = model.predict(roi, verbose=0)[m
[32m+[m[32m            emotion_index = int(np.argmax(prediction))[m
[32m+[m[32m            emotion = emotion_labels[emotion_index][m
[32m+[m[32m            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)[m
[32m+[m[32m        except Exception as e:[m
[32m+[m[32m            print(f"Error durante la predicción: {e}")[m
 [m
         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Dibuja un rectángulo alrededor del rostro detectado[m
 [m
