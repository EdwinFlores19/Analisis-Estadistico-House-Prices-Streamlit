# ------------------------------------------------------------------------------------
# 🏠_Página_Principal.py
# ------------------------------------------------------------------------------------

# Importamos las librerías necesarias
import streamlit as st
import pandas as pd
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Precios de Viviendas",
    page_icon="🏠",
    layout="wide"  # Usamos un layout ancho para aprovechar el espacio
)


# --- Funciones de Carga y Preprocesamiento de Datos ---
# Usamos el decorador @st.cache_data para que los datos se carguen una sola vez
@st.cache_data
def load_data(file_path):
    """
    Carga y preprocesa los datos desde un archivo CSV.
    - Convierte columnas categóricas ('yes'/'no') a valores numéricos (1/0).
    - Aplica One-Hot Encoding a la columna 'furnishingstatus'.
    """
    df = pd.read_csv(file_path)

    # Lista de columnas binarias (con valores 'yes' o 'no')
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

    # Mapeo de 'yes' a 1 y 'no' a 0
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    # Aplicar One-Hot Encoding a 'furnishingstatus'
    # Esto crea nuevas columnas para cada categoría (furnished, semi-furnished, etc.)
    df_encoded = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

    return df, df_encoded


# --- Cuerpo Principal de la Página ---

# Título y descripción
st.title("🏠 Dashboard de Análisis de Precios de Viviendas")
st.markdown("""
    ¡Bienvenido a este dashboard interactivo!

    Este proyecto utiliza un conjunto de datos sobre precios de viviendas para realizar análisis,
    visualizaciones y predicciones. Navega por las diferentes páginas usando el menú de la izquierda.

    **El objetivo es entender los factores que más influyen en el precio de una vivienda.**
""")

# Cargar los datos usando nuestra función
try:
    original_df, encoded_df = load_data('house-price-parquet.csv')

    # Sección para mostrar los datos
    st.header("Visualización del Conjunto de Datos")
    st.markdown("Aquí puedes ver una muestra de los datos originales:")
    st.dataframe(original_df.head(10))  # Mostramos las primeras 10 filas

    # Mostramos los datos después del preprocesamiento para que se entienda la transformación
    st.markdown("Y aquí los datos después de ser procesados para el análisis y el modelo:")
    st.dataframe(encoded_df.head(10))

    # Sección de estadísticas descriptivas
    st.header("Estadísticas Descriptivas")
    st.markdown("Resumen estadístico de las principales características numéricas:")
    # Usamos .describe() para obtener las estadísticas y lo mostramos en Streamlit
    stats = original_df[['price', 'area', 'bedrooms', 'bathrooms', 'stories']].describe()
    st.write(stats)

    st.info("""
        **Navegación:**
        - **Análisis Exploratorio:** Sumérgete en las visualizaciones de datos.
        - **Predicción de Precio:** Usa nuestro modelo para estimar el valor de una casa.
        - **Segmentación de Viviendas:** Descubre grupos naturales de viviendas basados en sus características.
    """)

except FileNotFoundError:
    st.error(
        "Error: No se encontró el archivo 'house-price-parquet.csv'. Asegúrate de que esté en la misma carpeta que el script.")
except Exception as e:
    st.error(f"Ocurrió un error al cargar los datos: {e}")