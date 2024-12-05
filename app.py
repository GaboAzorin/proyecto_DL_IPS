import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model

# Título de la aplicación
st.title("Predicciones de 'No Cobro' - Proyecto DL IPS-UAI - Marcela Zapata y Gabriel Azorín")

# Subir archivo CSV para predicción
uploaded_file = st.file_uploader("Sube tu archivo CSV para predicciones", type=["csv"])

# Cargar el modelo entrenado
@st.cache_resource
def load_model_keras():
    # Carga el modelo guardado con Keras
    model = load_model('modelo_entrenado.keras')  # Cambia el nombre si es necesario
    return model

# Cargar el modelo
model = load_model_keras()

# Lógica para manejar el archivo cargado
if uploaded_file is not None:
    # Leer el archivo cargado
    data = pd.read_csv(uploaded_file)

    # Mostrar el dataset cargado
    st.write("Dataset cargado:")
    st.write(data.head())

    # Asegurarse de que todas las columnas sean numéricas
    try:
        data = data.apply(pd.to_numeric, errors='coerce')

        # Verificar si el dataset contiene valores NaN
        if data.isnull().any().any():
            st.warning("Algunos valores no pudieron ser convertidos a numéricos. Por favor, verifica el archivo.")
            st.write("Dataset con valores NaN:")
            st.write(data)

        # Verificar si las columnas coinciden con lo que espera el modelo
        expected_columns = 12  # Cambia esto según el número de columnas que espera tu modelo
        if data.shape[1] != expected_columns:
            st.error(f"El archivo cargado no tiene el número correcto de columnas. Se esperaban {expected_columns}, pero se encontraron {data.shape[1]}.")
        else:
            # Realizar predicciones
            predictions = model.predict(data)

            # Convertir predicciones a clases binarias (0 o 1)
            predicciones_clases = [1 if pred > 0.5 else 0 for pred in predictions]

            # Mapear las predicciones a descripciones claras
            predicciones_descriptivas = ["Sí cobrará." if pred == 1 else "No cobrará, así que se sugiere contactar proactivamente." for pred in predicciones_clases]

            # Mostrar predicciones descriptivas
            st.write("Predicciones:")
            for i, prediccion in enumerate(predicciones_descriptivas):
                st.write(f"Persona nº {i}: {prediccion}")
    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")

