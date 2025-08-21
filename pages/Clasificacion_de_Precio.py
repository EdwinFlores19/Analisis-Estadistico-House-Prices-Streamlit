# ------------------------------------------------------------------------------------
# pages/7_✨_Clasificación_de_Precio.py
# ------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Reutilizamos la función de carga de datos
from Pagina_Principal import load_and_preprocess_data

st.set_page_config(
    page_title="Clasificación de Precio",
    page_icon="✨",
    layout="wide"
)

st.title("✨ Clasificación de Viviendas por Nivel de Precio")
st.markdown("""
    En esta página, cambiamos nuestro enfoque de predecir un precio exacto a
    **clasificar** las viviendas en diferentes niveles de precio.
    Utilizamos el modelo **K-Nearest Neighbors (KNN)** para categorizar una vivienda
    como 'Bajo', 'Medio' o 'Alto' en función de sus características.
""")

# Cargar y preprocesar los datos
_, encoded_df = load_and_preprocess_data('house-price-parquet.csv')

if encoded_df is not None:
    # Crear la variable objetivo de clasificación
    # Definimos los rangos de precios
    bins = [0, 4000000, 6000000, encoded_df['price'].max()]
    labels = ['Bajo', 'Medio', 'Alto']
    encoded_df['price_level'] = pd.cut(encoded_df['price'], bins=bins, labels=labels)

    # Definir características y la variable objetivo de clasificación
    features = [col for col in encoded_df.columns if col not in ['price', 'house_id', 'price_level']]
    X = encoded_df[features]
    y = encoded_df['price_level']

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.header("Rendimiento del Modelo de Clasificación")
    st.metric(label="Precisión del Modelo (Accuracy)", value=f"{accuracy:.2%}")
    st.info("La precisión es el porcentaje de predicciones correctas que hizo el modelo.")

    # --- Interfaz para la predicción del usuario ---
    st.sidebar.header("Predice el Nivel de Precio:")
    st.sidebar.markdown("Introduce las características para clasificar la vivienda.")

    # Widgets para entrada de datos (usamos los mismos que en la página de predicción)
    area = st.sidebar.slider('Área (pies cuadrados)', 1650, 16200, 5000)
    bedrooms = st.sidebar.slider('Número de Habitaciones', 1, 6, 3)
    bathrooms = st.sidebar.slider('Número de Baños', 1, 4, 2)
    stories = st.sidebar.slider('Número de Pisos', 1, 4, 2)
    parking = st.sidebar.slider('Plazas de Estacionamiento', 0, 3, 1)

    mainroad = st.sidebar.toggle('¿Está en la carretera principal?', value=True)
    guestroom = st.sidebar.toggle('¿Tiene cuarto de invitados?', value=False)
    basement = st.sidebar.toggle('¿Tiene sótano?', value=False)
    hotwater = st.sidebar.toggle('¿Tiene calefacción de agua?', value=False)
    airco = st.sidebar.toggle('¿Tiene aire acondicionado?', value=True)
    prefarea = st.sidebar.toggle('¿Está en un área preferencial?', value=False)
    furnishing = st.sidebar.selectbox(
        'Estado del Mobiliario',
        ('Amueblado', 'Semi-amueblado', 'Sin amueblar')
    )

    # Preparar datos de entrada para la predicción
    user_data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'mainroad': 1 if mainroad else 0,
        'guestroom': 1 if guestroom else 0,
        'basement': 1 if basement else 0,
        'hotwaterheating': 1 if hotwater else 0,
        'airconditioning': 1 if airco else 0,
        'parking': parking,
        'prefarea': 1 if prefarea else 0,
        'furnishingstatus_semi-furnished': 1 if furnishing == 'Semi-amueblado' else 0,
        'furnishingstatus_unfurnished': 1 if furnishing == 'Sin amueblar' else 0,
    }

    input_df = pd.DataFrame([user_data])
    input_df = input_df[features] # Reordenar las columnas

    # Realizar y mostrar la predicción
    prediction = knn_model.predict(input_df)

    st.header("Resultado de la Clasificación")
    st.info(f"El nivel de precio estimado para esta vivienda es: **{prediction[0]}**")
