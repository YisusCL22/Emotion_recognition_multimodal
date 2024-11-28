# emotion_detector.py
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import os

emotion_map = {0: 'Enojado', 1: 'Asco', 2: 'Miedo', 3: 'Feliz', 4: 'Triste', 5: 'Sorpresa', 6: 'Neutral'}
emotion_map_audio_model = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy', 4: 'neutral', 5: 'sadness', 6: 'surprise'}


class EmotionDetector:
    def __init__(self, model_path='facial_A77.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.frame_counter = 0

    def detect_faces(self, gray_frame):
        return self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    def predict_emotion(self, roi_gray):
        resized = cv2.resize(roi_gray, (48, 48))
        img_array = np.array(resized).reshape(1, 48, 48, 1) / 255.0
        predictions = self.model.predict(img_array)
        emotion_index = np.argmax(predictions)
        emotion = emotion_map.get(emotion_index, 'Desconocido')
        confidence = int(np.max(predictions) * 100)
        return emotion, confidence, predictions