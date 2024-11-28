from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Cargar el modelo y el tokenizador
model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Ver las etiquetas que el modelo utiliza para la clasificación
model_labels = model.config.id2label
print("Model labels:", model_labels)

# Texto a clasificar
text = "You should to do better!"

# Tokenizar el texto
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Realizar la predicción
with torch.no_grad():
    outputs = model(**inputs)

# Obtener las predicciones (probabilidades de las emociones)
logits = outputs.logits
probabilities = torch.nn.functional.softmax(logits, dim=-1)

# Mostrar todas las probabilidades para cada emoción
for idx, prob in enumerate(probabilities[0]):
    emotion = model_labels[idx]
    print(f"Emotion: {emotion}, Probability: {prob.item():.2f}")

# Obtener el índice de la clase con la mayor probabilidad
predicted_class_idx = torch.argmax(probabilities, dim=-1).item()

# Obtener la emoción predicha usando el índice
predicted_class = model_labels[predicted_class_idx]

# Mostrar la predicción y su probabilidad
predicted_probability = probabilities[0, predicted_class_idx].item()
print(f"\nPredicted Emotion: {predicted_class} with probability {predicted_probability:.2f}")
