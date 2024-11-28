from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import speech_recognition as sr
from deep_translator import GoogleTranslator

# Cargar el modelo y el tokenizador
model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Obtener etiquetas del modelo
model_labels = model.config.id2label

# Función para transcribir audio
def transcribir_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Por favor, habla ahora...")
        audio = recognizer.listen(source)
    try:
        texto = recognizer.recognize_google(audio, language="es-ES")
        print(f"Transcripción: {texto}")
        return texto
    except sr.UnknownValueError:
        print("No se pudo entender el audio.")
        return None
    except sr.RequestError as e:
        print(f"Error al conectarse al servicio de reconocimiento de voz: {e}")
        return None

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

# Función para predecir emociones
def predecir_emocion(modelo, tokenizer, texto):
    # Tokenizar el texto
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    # Realizar la predicción
    with torch.no_grad():
        outputs = modelo(**inputs)
    # Obtener probabilidades
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze()
    # Mapear etiquetas con sus probabilidades
    emociones_probabilidades = {model_labels[i]: prob.item() for i, prob in enumerate(probabilities)}
    # Ordenar las emociones por probabilidad
    emociones_ordenadas = sorted(emociones_probabilidades.items(), key=lambda item: item[1], reverse=True)
    
    # Mostrar emociones y probabilidades
    print("Emociones predichas:")
    for emocion, prob in emociones_ordenadas:
        print(f"- {emocion}: {prob * 100:.2f}%")
    
    # Devolver la emoción predominante
    return emociones_ordenadas[0][0]

# Menú principal
def ejecutar_prediccion():
    while True:
        print("\nSeleccione una opción:")
        print("1. Hablar y predecir emoción")
        print("2. Ingresar texto y predecir emoción")
        print("3. Salir")
        opcion = input("Ingrese el número de la opción deseada: ")
        
        if opcion == "1":
            texto_es = transcribir_audio()
            if texto_es:
                texto_en = traducir_texto(texto_es)
                if texto_en:
                    emocion = predecir_emocion(model, tokenizer, texto_en)
                    print(f"Emoción predominante: {emocion}")
        elif opcion == "2":
            texto_es = input("Ingrese el texto: ")
            if texto_es.strip():
                texto_en = traducir_texto(texto_es)
                if texto_en:
                    emocion = predecir_emocion(model, tokenizer, texto_en)
                    print(f"Emoción predominante: {emocion}")
            else:
                print("No se ingresó ningún texto.")
        elif opcion == "3":
            print("Programa terminado.")
            break
        else:
            print("Opción no válida. Por favor, ingrese '1', '2' o '3'.")

# Ejecutar el programa
if __name__ == "__main__":
    ejecutar_prediccion()
