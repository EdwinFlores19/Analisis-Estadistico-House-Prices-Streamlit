# ------------------------------------------------------------------------------------
# pages/6_🔑_Análisis_de_Características.py
# ------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Reutilizamos la función de carga de datos
from Pagina_Principal import load_and_preprocess_data

st.set_page_config(
    page_title="Análisis de Características",
    page_icon="🔑",
    layout="wide"
)

st.title("🔑 Análisis de Importancia de las Características")
st.markdown("""
    Aquí exploramos qué variables son las más importantes para predecir el precio de una vivienda.
    Utilizamos un modelo de **Random Forest** y visualizamos la importancia que asigna a cada característica.
""")

# Cargar los datos
_, encoded_df = load_and_preprocess_data('house-price-parquet.csv')

if encoded_df is not None:
    # Definir características (X) y variable objetivo (y)
    features = [col for col in encoded_df.columns if col not in ['price', 'house_id']]
    X = encoded_df[features]
    y = encoded_df['price']

    # Entrenar un modelo de Random Forest para obtener la importancia de las características
    # No es necesario dividir en train/test si solo queremos la importancia
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Obtener la importancia de las características
    feature_importances = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    # Mostrar la visualización
    st.header("Importancia Relativa de las Características")
    st.markdown("""
        Un valor más alto indica que la característica tiene una mayor influencia en la
        predicción del precio de la vivienda.
    """)

    # Gráfico de barras interactivo con Plotly
    fig = px.bar(
        feature_importances,
        x='importance',
        y='feature',
        orientation='h',
        title='Top Características más Importantes para la Predicción de Precios',
        labels={'importance': 'Importancia (Puntaje)', 'feature': 'Característica'}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # Mostrar la tabla de resultados
    st.header("Tabla de Importancia de Características")
    st.dataframe(feature_importances)

    st.info("""
        **Análisis de los resultados:** La `area` (área de la vivienda) es típicamente la característica más importante,
        seguida por otras variables como `bathrooms` y `stories`.
    """)
