# Importo librerias que se utilizaran
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Configuración de la página
st.set_page_config(
    page_title="Predicción Diabetes - Random Forest",
    layout="wide"
)

# Titulo y descripción de la app
st.title("Sistema de Predicción de Diabetes")
st.markdown("""
Esta aplicación utiliza un modelo de **Random Forest** para predecir la probabilidad de diabetes.
Por favor ingrese los datos clínicos del paciente en el menú lateral.
""")

# Carga del modelo
@st.cache_resource
def load_model():
    # Construir la ruta absoluta al modelo
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'streamlist-app-machine-learning', 'models', 'best_model.pkl')
    
    try:
        loaded_model = pickle.load(open(model_path, 'rb'))
        return loaded_model
    except FileNotFoundError:
        st.error(f"No se encontró el archivo del modelo en: {model_path}")
        return None

model = load_model()

# Sidebar para inputs del usuario
st.sidebar.header("Parámetros del Paciente")

def user_input_features():
    # Defino los inputs basandome en las columnas del dataset original
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 117)
    blood_pressure = st.sidebar.slider('BloodPressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('SkinThickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('DiabetesPedigreeFunction', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Age', 21, 81, 29)

    # Creo un diccionario con los datos
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    
    # Convierto a DataFrame (formato que espera el modelo)
    features = pd.DataFrame(data, index=[0])
    return features

# Capturo los datos del usuario
input_df = user_input_features()

# Muestro los datos ingresados en el panel principal
st.subheader('Datos Ingresados')
st.write(input_df)

# Botón para realizar la predicción
if st.button('Predecir Diabetes'):
    if model is not None:
        # Realizo la predicción
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader('Resultado de la Predicción')
        
        # Muestro el resultado
        if prediction[0] == 1:
            st.error(f'El modelo predice que el paciente **TIENE DIABETES**.')
        else:
            st.success(f'El modelo predice que el paciente **NO TIENE DIABETES**.')

        # Muestro la probabilidad
        st.write(f"Probabilidad de Diabetes: {prediction_proba[0][1]*100:.2f}%")
        st.write(f"Probabilidad de No Diabetes: {prediction_proba[0][0]*100:.2f}%")
        
    else:
        st.warning("El modelo no está cargado, verifica el archivo .pkl")

# Footer
st.markdown("---")
st.markdown("Desarrollado con Random Forest y Streamlit")