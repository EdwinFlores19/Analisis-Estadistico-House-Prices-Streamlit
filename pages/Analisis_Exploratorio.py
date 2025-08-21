# ------------------------------------------------------------------------------------
# pages/2_📊_Análisis_Exploratorio.py
# ------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff


# Usamos la misma función de carga de datos para consistencia
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    df_encoded = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)
    return df, df_encoded


# Cargamos los datos
try:
    original_df, encoded_df = load_data('house-price-parquet.csv')

    st.title("📊 Análisis Exploratorio de Datos (EDA)")
    st.markdown("En esta sección, visualizamos los datos para descubrir patrones y relaciones.")

    # --- 1. Distribución de Precios ---
    st.header("1. Distribución del Precio de las Viviendas")
    st.markdown(
        "El histograma nos muestra cómo se distribuyen los precios. La mayoría de las casas se concentran en un rango de precios más bajo.")

    # Creamos un histograma interactivo con Plotly
    fig_hist = px.histogram(original_df, x='price', nbins=50, title="Distribución de Precios",
                            labels={'price': 'Precio (en millones)'})
    fig_hist.update_layout(bargap=0.1)
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- 2. Diagrama de Dispersión: Área vs. Precio ---
    st.header("2. Relación entre Área y Precio")
    st.markdown(
        "Este diagrama de dispersión es clave: muestra que, como es de esperar, **a mayor área, mayor es el precio**.")

    # Widget para seleccionar el número de habitaciones y filtrar el gráfico
    st.sidebar.header("Filtros para el Gráfico de Dispersión")
    bedroom_filter = st.sidebar.multiselect(
        'Filtrar por número de habitaciones:',
        options=sorted(original_df['bedrooms'].unique()),
        default=sorted(original_df['bedrooms'].unique())
    )

    # Filtramos el dataframe según la selección del usuario
    filtered_df = original_df[original_df['bedrooms'].isin(bedroom_filter)]

    # Creamos un diagrama de dispersión interactivo con Plotly
    # Añadimos color por número de habitaciones para más detalle
    fig_scatter = px.scatter(filtered_df, x='area', y='price',
                             color='bedrooms',
                             title="Área vs. Precio de la Vivienda",
                             labels={'area': 'Área (pies cuadrados)', 'price': 'Precio'},
                             hover_data=['stories', 'bathrooms'])  # Datos extra al pasar el cursor
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- 3. Matriz de Correlación ---
    st.header("3. Correlación entre Características")
    st.markdown("""
        El mapa de calor nos muestra la correlación de Pearson entre las variables numéricas.
        - Un valor cercano a **1 (azul oscuro)** indica una fuerte correlación positiva.
        - Un valor cercano a **-1 (rojo)** indica una fuerte correlación negativa.
        - Un valor cercano a **0 (blanco)** indica poca o ninguna correlación lineal.

        Observamos fuertes correlaciones positivas entre `price` y `area`, `bathrooms`, y `stories`.
    """)

    # Calculamos la matriz de correlación de los datos procesados
    corr_matrix = encoded_df.corr()

    # Creamos el mapa de calor con Plotly
    fig_heatmap = px.imshow(corr_matrix,
                            text_auto=True,  # Mostrar los valores numéricos
                            aspect="auto",
                            color_continuous_scale='RdBu_r',  # Paleta de colores invertida
                            title="Mapa de Calor de Correlaciones")
    st.plotly_chart(fig_heatmap, use_container_width=True)


except FileNotFoundError:
    st.error(
        "No se pudo encontrar el archivo de datos. Asegúrate de que 'house-price-parquet.csv' esté en el directorio principal.")
except Exception as e:
    st.error(f"Ocurrió un error: {e}")