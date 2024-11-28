# text_emotion_detector.py
from transformers import pipeline

class TextEmotionDetector:
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        self.classifier = pipeline("text-classification", model=model_name, return_all_scores=True)

    def predict_emotion(self, text):
        predictions = self.classifier(text)
        emotion_scores = predictions[0]
        emotion_scores.sort(key=lambda x: x['score'], reverse=True)
        top_emotion = emotion_scores[0]['label']
        confidence = int(emotion_scores[0]['score'] * 100)
        return top_emotion, confidence, emotion_scores


"""
# Ejemplo de uso
if __name__ == "__main__":
    detector = TextEmotionDetector()
    emotion, confidence, scores = detector.predict_emotion("I love this!")
    print(f"Emotion: {emotion}, Confidence: {confidence}%")
    print("All scores:", scores)
"""