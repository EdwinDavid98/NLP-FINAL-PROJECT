import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import json
import random  # Importar random para hacer la selección de respuestas aleatorias

# Definir el path donde están los archivos guardados
path = '/home/emontenegrob/Labs_NLP/prueba2_chatbot/roberta_complete_model/'

# Cargar el tokenizador y el modelo guardado
tokenizer = RobertaTokenizer.from_pretrained(path)
model = RobertaForSequenceClassification.from_pretrained(path)

# Mover el modelo a la GPU si está disponible
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()  # Establecer el modo de evaluación

# Cargar las etiquetas del dataset original (las intenciones)
with open('/home/emontenegrob/Labs_NLP/prueba2_chatbot/intents_universidad2.json', 'r') as file:
    data = json.load(file)

# Obtener el mapeo de etiquetas (intenciones)
labels = [intent['tag'] for intent in data['intents']]

# Función para predecir la intención del usuario
def predict_intent(text):
    # Tokenizar la entrada del usuario
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=64)

    # Mover los inputs a la GPU si está disponible
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Hacer la predicción
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()

    # Obtener la etiqueta (intención) correspondiente
    predicted_label = labels[predicted_class_id]
    return predicted_label

# Función para obtener una respuesta basada en la intención
def get_response(intent):
    for i in data['intents']:
        if i['tag'] == intent:
            # Usar random.choice para elegir una respuesta aleatoria
            return random.choice(i['responses'])

# Interfaz para interactuar con el modelo
if __name__ == "__main__":
    print("Chatbot está listo para recibir preguntas. Escribe 'salir' para terminar.")

    while True:
        # Leer la entrada del usuario
        user_input = input("Escribe tu pregunta: ")

        if user_input.lower() == "salir":
            print("¡Hasta luego!")
            break

        # Predecir la intención del usuario
        intent = predict_intent(user_input)
        print(f"Intento predicho: {intent}")

        # Obtener la respuesta correspondiente
        response = get_response(intent)
        print(f"Respuesta del bot: {response}")

