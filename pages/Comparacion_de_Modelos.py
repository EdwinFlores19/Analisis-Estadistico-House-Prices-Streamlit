# ------------------------------------------------------------------------------------
# pages/8_📊_Comparación_de_Modelos.py
# ------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Reutilizamos la función de carga de datos
from Pagina_Principal import load_and_preprocess_data

st.set_page_config(
    page_title="Comparación de Modelos",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Comparación de Modelos de Predicción")
st.markdown("""
    En esta página, comparamos el rendimiento de diferentes modelos de Machine Learning
    para la predicción del precio de las viviendas.
    Evaluamos la **Regresión Lineal**, **Random Forest** y **Gradient Boosting**.
""")

# Cargar los datos
_, encoded_df = load_and_preprocess_data('house-price-parquet.csv')

if encoded_df is not None:
    # Definir características y variable objetivo
    features = [col for col in encoded_df.columns if col not in ['price', 'house_id']]
    X = encoded_df[features]
    y = encoded_df['price']

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Diccionario de modelos a comparar
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    # Entrenar y evaluar cada modelo
    results = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        results.append({
            'Modelo': model_name,
            'R² Score': r2,
            'MAE (Mean Absolute Error)': mae,
            'MSE (Mean Squared Error)': mse
        })

    results_df = pd.DataFrame(results)

    st.header("Resultados de la Comparación de Modelos")
    st.markdown("""
        Observa cómo cada modelo se desempeña en diferentes métricas:
        - **R² Score:** Indica la proporción de la varianza en la variable dependiente que es predecible a partir de las variables independientes. Un valor más cercano a 1 es mejor.
        - **MAE (Error Absoluto Medio):** Mide la magnitud promedio de los errores en un conjunto de predicciones, sin considerar su dirección.
        - **MSE (Error Cuadrático Medio):** Similar al MAE, pero penaliza más los errores grandes.
    """)
    st.dataframe(results_df.set_index('Modelo'))

    st.header("Visualización de las Métricas")
    st.markdown("Gráfico de barras que compara el R² Score de cada modelo.")
    fig = px.bar(
        results_df,
        x='Modelo',
        y='R² Score',
        title='Comparación de R² Score entre Modelos',
        color='Modelo'
    )
    st.plotly_chart(fig, use_container_width=True)
