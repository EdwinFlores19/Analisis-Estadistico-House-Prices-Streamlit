# ------------------------------------------------------------------------------------
# pages/4_🧩_Segmentación_de_Viviendas.py
# ------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# --- Funciones de Carga y Clustering ---
@st.cache_data
def load_data(file_path):
    """Carga los datos originales."""
    df = pd.read_csv(file_path)
    return df


@st.cache_data
def perform_clustering(df, n_clusters):
    """
    Realiza clustering K-Means sobre las características 'price' y 'area'.
    Devuelve el dataframe con una nueva columna 'cluster'.
    """
    # Seleccionamos las características para el clustering
    features = df[['price', 'area']]

    # Es fundamental escalar los datos para que K-Means funcione correctamente
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Creamos y entrenamos el modelo K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(features_scaled)

    # Añadimos la etiqueta del clúster al dataframe original
    df['cluster'] = kmeans.labels_
    return df


# --- Interfaz de la Página de Clustering ---
st.title("🧩 Segmentación de Viviendas (Clustering K-Means)")
st.markdown("""
    En esta página aplicamos una técnica de Machine Learning no supervisado para
    encontrar grupos o **segmentos naturales** de viviendas en los datos.

    Usamos el algoritmo **K-Means** basándonos en el `precio` y el `área` de las viviendas.
    Puedes ajustar el número de segmentos a encontrar con el slider.
""")

try:
    df_original = load_data('house-price-parquet.csv')

    # Slider para que el usuario elija el número de clústers
    st.sidebar.header("Configuración del Clustering")
    num_clusters = st.sidebar.slider(
        'Selecciona el número de segmentos (K):',
        min_value=2,
        max_value=8,
        value=4  # Valor por defecto
    )

    # Realizar el clustering
    df_clustered = perform_clustering(df_original.copy(), num_clusters)

    # --- Visualización de los Clústers ---
    st.header(f"Visualización de {num_clusters} Segmentos de Viviendas")

    # Creamos un gráfico de dispersión interactivo coloreado por clúster
    fig_cluster = px.scatter(
        df_clustered,
        x='area',
        y='price',
        color='cluster',
        color_continuous_scale=px.colors.qualitative.Vivid,
        title='Segmentos de Viviendas por Área y Precio',
        labels={'area': 'Área (pies cuadrados)', 'price': 'Precio'},
        hover_data=['bedrooms', 'bathrooms']
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    # --- Análisis de los Clústers ---
    st.header("Análisis de las Características por Segmento")
    st.markdown(
        "A continuación, se muestra el promedio de las características principales para cada segmento encontrado.")

    # Calculamos el promedio de las características numéricas para cada clúster
    cluster_analysis = df_clustered.groupby('cluster')[['price', 'area', 'bedrooms', 'bathrooms']].mean().reset_index()
    st.dataframe(cluster_analysis)

    st.info("""
        **¿Cómo interpretar esto?**
        - El **Clúster 0** podría representar viviendas de bajo costo y área reducida.
        - Otro clúster podría agrupar las viviendas más lujosas y espaciosas.

        Esta técnica es muy útil en marketing para dirigir campañas a segmentos específicos de clientes.
    """)

except FileNotFoundError:
    st.error("No se pudo encontrar el archivo de datos.")
except Exception as e:
    st.error(f"Ocurrió un error: {e}")