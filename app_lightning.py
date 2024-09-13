import streamlit as st
import time
from model_lightning import predict_intent, get_response  # Importamos las funciones de predicción

# Configuración de la página
st.set_page_config(page_title="UNACH BOT", page_icon="📲", layout="centered")

# Estilo personalizado con tema oscuro
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stTextInput > div > div > input {
        background-color: #2E2E2E;
        color: #FFFFFF;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)

# Título y descripción
st.title('📲 UNACH BOT Asistente Virtual')
st.markdown("Bienvenido al asistente virtual de la Universidad Nacional de Chimborazo. ¿En qué puedo ayudarte hoy?")

# Inicializar la lista de mensajes si no existe
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hola, soy el asistente virtual de la UNACH. ¿En qué puedo ayudarte?"}
    ]

# Función para añadir mensajes
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

# Mostrar los mensajes
for message in st.session_state.messages[-50:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada del usuario
user_input = st.chat_input("Escribe tu pregunta aquí...")

# Procesar la entrada del usuario
if user_input:
    add_message("user", user_input)

    # Mostrar "Escribiendo..." del chatbot
    with st.chat_message("assistant"):
        st.markdown("Escribiendo...")
        time.sleep(1.5)

    # Predecir la intención del mensaje del usuario y obtener la respuesta
    intent_prediction = predict_intent(user_input)  # Usamos la función de predicción
    response = get_response(intent_prediction)  # Obtenemos la respuesta

    # Añadir la respuesta del asistente
    add_message("assistant", response)
    st.rerun()

# Opción para borrar el historial de chat
if st.button("Borrar historial de chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hola, soy el asistente virtual de la UNACH. ¿En qué puedo ayudarte?"}
    ]
    st.rerun()

# Pie de página
st.markdown("---")
st.markdown("Desarrollado por el equipo de TI de la UNACH")
