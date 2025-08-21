# ------------------------------------------------------------------------------------
# pages/5_🕵️_Detección_de_Outliers.py
# ------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
import numpy as np

# Reutilizamos la función de carga de datos desde la página principal
from Pagina_Principal import load_and_preprocess_data

st.set_page_config(
    page_title="Detección de Outliers",
    page_icon="🕵️",
    layout="wide"
)

st.title("🕵️ Detección de Valores Atípicos (Outliers)")
st.markdown("""
    Esta página utiliza el algoritmo **Isolation Forest** para identificar viviendas
    que, por sus características, se desvían significativamente del resto del conjunto de datos.
    Estas son candidatas a ser **valores atípicos**.
""")

# Cargar los datos
original_df, _ = load_and_preprocess_data('house-price-parquet.csv')

if original_df is not None:
    # Seleccionar características para el modelo. Excluimos las binarias y categóricas.
    features = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    data_for_model = original_df[features]

    # Configuración de Isolation Forest
    # El 'contamination' es el porcentaje estimado de outliers en los datos.
    model = IsolationForest(contamination=0.05, random_state=42)

    # Entrenar el modelo
    model.fit(data_for_model)

    # Predecir los outliers. -1 significa outlier, 1 significa dato normal.
    outlier_labels = model.predict(data_for_model)
    # Obtener la puntuación de anomalía. Valores más bajos (más negativos) indican mayor probabilidad de ser un outlier.
    anomaly_scores = model.decision_function(data_for_model)

    # Corregir el error: Transformamos las puntuaciones negativas a valores positivos para el tamaño de los puntos
    # Invertimos el signo y sumamos el valor máximo para que las puntuaciones más negativas
    # se conviertan en los valores positivos más grandes.
    max_score = anomaly_scores.max()
    transformed_scores = max_score - anomaly_scores

    # Añadir los resultados al DataFrame original
    original_df['is_outlier'] = [1 if label == -1 else 0 for label in outlier_labels]
    original_df['anomaly_score'] = transformed_scores

    # Mostrar la visualización
    st.header("Visualización de Valores Atípicos")
    st.markdown("""
        El gráfico de dispersión muestra las viviendas. Las que se han identificado como
        **atípicas** están marcadas en rojo. El tamaño del punto refleja su `anomaly_score`:
        los puntos más grandes son los que el modelo considera más atípicos.
    """)

    # Gráfico interactivo con Plotly para visualizar los outliers
    fig = px.scatter(
        original_df,
        x='area',
        y='price',
        color='is_outlier',
        color_discrete_map={0: 'blue', 1: 'red'},
        size='anomaly_score',
        size_max=15,
        title="Detección de Outliers (Área vs Precio)",
        labels={'is_outlier': 'Es Outlier', 'area': 'Área', 'price': 'Precio'},
        hover_data=['bedrooms', 'bathrooms', 'stories']
    )
    st.plotly_chart(fig, use_container_width=True)

    # Mostrar la tabla con los outliers
    st.header("Datos de las Viviendas Atípicas")
    st.markdown("Aquí puedes ver los detalles de las viviendas identificadas como atípicas:")
    outliers_df = original_df[original_df['is_outlier'] == 1].sort_values(by='anomaly_score', ascending=False)
    st.dataframe(outliers_df)

    st.info("""
        **Nota:** Los valores atípicos no son necesariamente "errores" en los datos. Podrían
        ser viviendas únicas o de lujo con características que las distinguen del resto.
    """)
