import json
import nlpaug.augmenter.word as naw

# Cargar el dataset JSON original
with open('/home/emontenegrob/Labs_NLP/prueba2_chatbot/intents_universidad2.json', 'r') as file:
    data = json.load(file)

# Inicializar el augmentador de sinónimos con una probabilidad menor y más restricciones
aug = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=1, aug_p=0.1)

# Definir cuántos ejemplos adicionales quieres generar por cada patrón original
num_augmented_examples = 3  # Ajusta según tus necesidades

def es_coherente(texto):
    """Función simple para evitar palabras incorrectas o incoherentes"""
    palabras_incoherentes = ["pine tree state", "atomic", "mile", "number"]
    return all(palabra not in texto.lower() for palabra in palabras_incoherentes)

# Para cada intención, aumentar los patrones de entrada controlando la coherencia
for intent in data['intents']:
    original_patterns = intent['patterns'][:]
    
    # Aplicar augmentación sobre cada patrón original
    for pattern in original_patterns:
        for _ in range(num_augmented_examples):
            augmented_pattern = aug.augment(pattern)
            
            # Verificar que el patrón generado sea coherente y no repetido
            if isinstance(augmented_pattern, list):
                augmented_pattern = augmented_pattern[0]
                
            if es_coherente(augmented_pattern) and augmented_pattern != pattern and augmented_pattern not in intent['patterns']:
                intent['patterns'].append(augmented_pattern)

# Guardar el dataset aumentado en un nuevo archivo
with open('/home/emontenegrob/Labs_NLP/prueba2_chatbot/intents_universidad2_augmented_clean.json', 'w') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print("Dataset aumentado y filtrado guardado en 'intents_universidad2_augmented.json'.")

