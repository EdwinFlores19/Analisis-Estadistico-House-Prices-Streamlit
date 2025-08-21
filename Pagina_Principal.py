# ------------------------------------------------------------------------------------
# 🏠_Página_Principal.py
# ------------------------------------------------------------------------------------

# Importamos las librerías necesarias
import streamlit as st
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Precios de Viviendas",
    page_icon="🏠",
    layout="wide"
)

# --- Funciones de Carga y Preprocesamiento de Datos ---
# Usamos el decorador @st.cache_data para que los datos se carguen una sola vez
@st.cache_data
def load_and_preprocess_data(file_path):
    """
    Carga y preprocesa los datos desde un archivo CSV.
    - Convierte columnas categóricas ('yes'/'no') a valores numéricos (1/0).
    - Aplica One-Hot Encoding a la columna 'furnishingstatus'.
    """
    try:
        df = pd.read_csv(file_path)

        # Mapeo de columnas binarias ('yes'/'no' a 1/0)
        binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
        for col in binary_cols:
            df[col] = df[col].map({'yes': 1, 'no': 0})

        # Aplicar One-Hot Encoding a 'furnishingstatus'
        df_encoded = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

        return df, df_encoded
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo de datos en la ruta especificada: {file_path}")
        return None, None
    except Exception as e:
        st.error(f"Ocurrió un error al cargar o preprocesar los datos: {e}")
        return None, None


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
original_df, encoded_df = load_and_preprocess_data('house-price-parquet.csv')

if original_df is not None and encoded_df is not None:
    # Sección para mostrar los datos
    st.header("Visualización del Conjunto de Datos")
    st.markdown("Aquí puedes ver una muestra de los datos originales:")
    st.dataframe(original_df.head(10))

    # Mostramos los datos después del preprocesamiento
    st.markdown("Y aquí los datos después de ser procesados para el análisis y el modelo:")
    st.dataframe(encoded_df.head(10))

    # Sección de estadísticas descriptivas
    st.header("Estadísticas Descriptivas")
    st.markdown("Resumen estadístico de las principales características numéricas:")
    stats = original_df[['price', 'area', 'bedrooms', 'bathrooms', 'stories']].describe()
    st.write(stats)

    st.info("""
        **Navegación:**
        - **Análisis Exploratorio:** Sumérgete en las visualizaciones de datos.
        - **Predicción de Precio:** Usa nuestro modelo para estimar el valor de una casa.
        - **Segmentación de Viviendas:** Descubre grupos naturales de viviendas.
        - **Nuevas páginas disponibles:**
            - **Detección de Outliers**
            - **Análisis de Características**
            - **Clasificación de Precio**
            - **Comparación de Modelos**
            - **Simulación de Escenarios**
    """)
