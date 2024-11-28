import cv2
import numpy as np
import tensorflow as tf
import speech_recognition as sr
import threading
import time
import torch
import queue
import unicodedata
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from deep_translator import GoogleTranslator

#Cargar Modelos
model = tf.keras.models.load_model('facial_A77.h5')
emotion_map = {0: 'Enojado', 1: 'Asco', 2: 'Miedo', 3: 'Feliz', 4: 'Triste', 5: 'Sorpresa', 6: 'Neutral'}
# Cargar el modelo y el tokenizador
model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_audio = AutoModelForSequenceClassification.from_pretrained(model_name)

# Ver las etiquetas que el modelo utiliza para la clasificación
model_labels = model_audio.config.id2label
print("Model labels:", model_labels)

#Función para GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# text_model.to(device)

cap = cv2.VideoCapture(0)

#Detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Configuración para la captura de audio
recognizer = sr.Recognizer()
audio_text = ""
audio_lock = threading.Lock()

predicted_emotion = ""
predicted_probability = 0

running = True 

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_thickness = 1

# Colas para sincronizar los datos
video_queue = queue.Queue()
audio_queue = queue.Queue()
text_queue = queue.Queue()

# Función para traducir texto al inglés
def traducir_texto(texto):
    try:
        traductor = GoogleTranslator(source="es", target="en")
        texto_traducido = traductor.translate(texto)
        print(f"Texto traducido: {texto_traducido}")
        return texto_traducido
    except Exception as e:
        print(f"Error al traducir el texto: {e}")
        return None

def limpiar_texto(texto):
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore').decode('utf-8')  # Eliminar caracteres no ASCII
    return texto

def capture_text():
    global audio_text, predicted_emotion
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        while running:
            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=3)
                partial_text = recognizer.recognize_google(audio, language="es-ES")
                partial_text = limpiar_texto(partial_text)
                with audio_lock:
                    audio_text = partial_text
                print("Texto reconocido:", partial_text)
                #audio_queue.put(partial_text)
                #Tokenizar y predecir la emoción del texto
                texto = traducir_texto(partial_text)
                inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
                # Realizar la predicción
                with torch.no_grad():
                    outputs = model_audio(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
                    predicted_emotion = model_labels[predicted_class_idx]
                    predicted_probability = probabilities[0, predicted_class_idx].item()
            except sr.UnknownValueError:
                with audio_lock:
                    audio_text = "No se logra captar el audio"
                    audio_queue.put(audio_text)
            except sr.RequestError:
                with audio_lock:
                    audio_text = "Error"
                    audio_queue.put(audio_text)
            except Exception as e:
                print(f"Error al capturar audio: {e}")
            time.sleep(0.1)

def capture_video():
    global audio_text
    while True:
        ret, frame = cap.read()  # Leer un frame de la cámara
        if not ret:
            break

        # Convertir el frame a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros en el frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Procesar las caras detectadas
        for (x, y, w, h) in faces:
            # Extraer la región de interés (ROI) de la cara
            roi_gray = gray[y:y+h, x:x+w]
            resized = cv2.resize(roi_gray, (48, 48))

            # Preprocesar la imagen
            img_array = np.array(resized).reshape(1, 48, 48, 1) / 255.0

            # Hacer una predicción
            predictions = model.predict(img_array)
            emotion_index = np.argmax(predictions)
            emotion = emotion_map[emotion_index]

            # Dibujar un rectángulo alrededor de la cara
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Agregar el texto de la emoción al frame
            text_emotion = f'Emocion: {emotion}'
            cv2.putText(frame, text_emotion, (x, y-10), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Mostrar el texto de audio capturado
        with audio_lock:
            cv2.putText(frame, f'Texto: {audio_text, predicted_emotion}', (10, 80), font, font_scale, (0, 255, 0), 2, cv2.LINE_AA)

        # Mostrar el frame en una ventana
        cv2.imshow('Video en tiempo real', frame)
        #video_queue.put(emotion)
        # Salir del bucle si presionas 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#Hilos para captura de audio y video
text_thread = threading.Thread(target=capture_text, daemon=True)
video_thread = threading.Thread(target=capture_video)

#Iniciar los hilos
text_thread.start()
video_thread.start()

try:
    video_thread.join()
except KeyboardInterrupt:
    print("Interrumpido por el usuario")


running = False


cap.release()
cv2.destroyAllWindows()
