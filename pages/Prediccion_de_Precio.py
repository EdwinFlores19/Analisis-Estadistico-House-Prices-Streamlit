# ------------------------------------------------------------------------------------
# pages/3_📈_Predicción_de_Precio.py
# ------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np


# --- Funciones de Carga y Entrenamiento del Modelo ---
@st.cache_data
def load_data(file_path):
    """Carga y preprocesa los datos."""
    df = pd.read_csv(file_path)
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    df_encoded = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)
    return df_encoded


@st.cache_resource  # Usamos cache_resource para objetos complejos como modelos
def train_model(df):
    """Entrena un modelo de Regresión Lineal y devuelve el modelo y su R² score."""
    # Definimos las características (X) y la variable objetivo (y)
    features = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement',
                'hotwaterheating', 'airconditioning', 'parking', 'prefarea',
                'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']
    X = df[features]
    y = df['price']

    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creamos y entrenamos el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluamos el modelo
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)

    return model, score, features


# --- Interfaz de la Página de Predicción ---
st.title("📈 Predicción del Precio de la Vivienda")
st.markdown("""
    Utiliza los controles en la barra lateral para introducir las características
    de una vivienda y nuestro modelo de Machine Learning estimará su precio.
""")

try:
    # Cargar datos y entrenar el modelo
    df_encoded = load_data('house-price-parquet.csv')
    model, r2_score_value, feature_names = train_model(df_encoded)

    # Mostrar el rendimiento del modelo
    st.write(f"**Rendimiento del Modelo (R² Score):** `{r2_score_value:.2f}`")
    st.info("Un R² Score cercano a 1 indica que el modelo explica una gran parte de la variabilidad del precio.")

    # --- Barra Lateral con los Controles de Usuario ---
    st.sidebar.header("Introduce las Características de la Vivienda:")

    # Sliders para características numéricas
    area = st.sidebar.slider('Área (pies cuadrados)', 1650, 16200, 5000)
    bedrooms = st.sidebar.slider('Número de Habitaciones', 1, 6, 3)
    bathrooms = st.sidebar.slider('Número de Baños', 1, 4, 2)
    stories = st.sidebar.slider('Número de Pisos', 1, 4, 2)
    parking = st.sidebar.slider('Plazas de Estacionamiento', 0, 3, 1)

    # Selectores y toggles para características categóricas
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

    # --- Preparación de los datos de entrada para la predicción ---
    # Creamos un diccionario con los datos del usuario
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

    # Convertimos el diccionario a un DataFrame de una sola fila
    input_df = pd.DataFrame([user_data])

    # Reordenamos las columnas para que coincidan con las del entrenamiento
    input_df = input_df[feature_names]

    # --- Realizar y Mostrar la Predicción ---
    prediction = model.predict(input_df)

    st.header("Resultado de la Predicción")
    # Usamos st.metric para un display más atractivo
    st.metric(label="Precio Estimado de la Vivienda", value=f"${prediction[0]:,.0f}")

    # Expander para mostrar detalles avanzados (opcional pero muy útil)
    with st.expander("Ver detalles de la entrada y el modelo"):
        st.write("Valores de entrada seleccionados:")
        st.dataframe(input_df)
        st.write("Coeficientes del modelo de Regresión Lineal:")
        # Mostramos la importancia de cada característica para el modelo
        coefs = pd.DataFrame(model.coef_, index=feature_names, columns=['Coeficiente'])
        st.dataframe(coefs.sort_values(by='Coeficiente', ascending=False))

except FileNotFoundError:
    st.error("No se pudo encontrar el archivo de datos.")
except Exception as e:
    st.error(f"Ocurrió un error: {e}")