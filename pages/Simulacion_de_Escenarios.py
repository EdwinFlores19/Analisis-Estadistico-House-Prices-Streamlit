# ------------------------------------------------------------------------------------
# pages/9_🔮_Simulación_de_Escenarios.py
# ------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Reutilizamos la función de carga de datos y entrenamiento del modelo
from Pagina_Principal import load_and_preprocess_data
from pages.Prediccion_de_Precio import train_model

st.set_page_config(
    page_title="Simulación de Escenarios",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Simulador de Precios (What-If)")
st.markdown("""
    Explora cómo un cambio en una sola característica puede afectar el precio
    predicho de una vivienda.
    Utiliza los controles en la barra lateral para ajustar el valor de una característica
    y observa el impacto inmediato en el precio.
""")

# Cargar los datos y entrenar el modelo (reutilizamos la función de la página de predicción)
_, encoded_df = load_and_preprocess_data('house-price-parquet.csv')
if encoded_df is not None:
    model, _, feature_names = train_model(encoded_df)

    st.info("El modelo base de este simulador es la Regresión Lineal, con un R² de 0.67.")

    # --- Barra lateral para la configuración ---
    st.sidebar.header("Configuración del Escenario")

    # Sliders para las características más importantes
    st.sidebar.markdown("Define un escenario base:")
    base_area = st.sidebar.slider('Área Base (pies cuadrados)', 1650, 16200, 5000, key='base_area')
    base_bedrooms = st.sidebar.slider('Habitaciones Base', 1, 6, 3, key='base_bedrooms')
    base_bathrooms = st.sidebar.slider('Baños Base', 1, 4, 2, key='base_bathrooms')
    base_stories = st.sidebar.slider('Pisos Base', 1, 4, 2, key='base_stories')
    base_parking = st.sidebar.slider('Parking Base', 0, 3, 1, key='base_parking')

    # Diccionario con los valores base para la simulación
    base_values = {
        'area': base_area, 'bedrooms': base_bedrooms, 'bathrooms': base_bathrooms,
        'stories': base_stories, 'parking': base_parking,
        'mainroad': 1, 'guestroom': 0, 'basement': 0, 'hotwaterheating': 0,
        'airconditioning': 1, 'prefarea': 0, 'furnishingstatus_semi-furnished': 0,
        'furnishingstatus_unfurnished': 0
    }

    # --- Simulación "What-If" ---
    st.header("Simulación paso a paso")
    st.markdown("Modifica solo una variable a la vez para ver el impacto:")

    # Opción para seleccionar la característica a cambiar
    feature_to_change = st.selectbox(
        'Elige la característica para simular el cambio:',
        options=['area', 'bedrooms', 'bathrooms', 'stories', 'parking',
                 'airconditioning', 'prefarea']
    )

    # Slider dinámico para la característica seleccionada
    if feature_to_change == 'area':
        new_value = st.slider('Nueva Área', 1650, 16200, base_area)
    elif feature_to_change == 'bedrooms':
        new_value = st.slider('Nuevas Habitaciones', 1, 6, base_bedrooms)
    elif feature_to_change == 'bathrooms':
        new_value = st.slider('Nuevos Baños', 1, 4, base_bathrooms)
    elif feature_to_change == 'stories':
        new_value = st.slider('Nuevos Pisos', 1, 4, base_stories)
    elif feature_to_change == 'parking':
        new_value = st.slider('Nuevo Parking', 0, 3, base_parking)
    elif feature_to_change in ['airconditioning', 'prefarea']:
        new_value = 1 if st.checkbox(f"¿Tiene {feature_to_change}?", value=bool(base_values[feature_to_change])) else 0
    else:
        new_value = base_values[feature_to_change]

    # Predecir el precio base y el nuevo precio
    base_input_df = pd.DataFrame([base_values])[feature_names]
    base_prediction = model.predict(base_input_df)[0]

    # Crear el nuevo escenario de simulación
    scenario_values = base_values.copy()
    scenario_values[feature_to_change] = new_value
    scenario_input_df = pd.DataFrame([scenario_values])[feature_names]
    scenario_prediction = model.predict(scenario_input_df)[0]

    st.header("Resultados de la Simulación")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Precio Base Estimado", value=f"${base_prediction:,.0f}")
        st.caption(f"Con {feature_to_change}: {base_values[feature_to_change]}")

    with col2:
        st.metric(
            label="Precio del Nuevo Escenario",
            value=f"${scenario_prediction:,.0f}",
            delta=f"${scenario_prediction - base_prediction:,.0f}"
        )
        st.caption(f"Con {feature_to_change} modificado a: {new_value}")
