import streamlit as st
import pandas as pd
import pickle

# Título de la aplicación
st.title("Predicciones de 'No Cobro' - Proyecto DL IPS-UAI - Marcela Zapata y Gabriel Azorín")

# Subir archivo CSV para predicción
uploaded_file = st.file_uploader("Sube tu archivo CSV para predicciones", type=["csv"])

# Cargar el modelo entrenado
@st.cache
def load_model():
    with open('modelo_entrenado.keras', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

if uploaded_file is not None:
    # Leer el archivo cargado
    data = pd.read_csv(uploaded_file)

    # Mostrar el dataset cargado
    st.write("Dataset cargado:")
    st.write(data.head())

    # Realizar predicciones
    predictions = model.predict(data)

    # Mostrar predicciones
    st.write("Predicciones:")
    st.write(predictions)

# Guardar el modelo entrenado
with open('modelo_entrenado.keras', 'wb') as file:
    pickle.dump(trained_model, file)  # Reemplaza 'trained_model' con el nombre de tu modelo