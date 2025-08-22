🏠 Análisis Estadístico de Precios de Viviendas con Streamlit
Visión General
Este proyecto es una aplicación web interactiva construida con Streamlit para realizar un análisis estadístico y predictivo sobre un conjunto de datos de precios de viviendas. El objetivo principal es explorar las características que influyen en el valor de las casas y demostrar diversas técnicas de Machine Learning, desde la visualización de datos hasta la predicción y el análisis de modelos.

La aplicación está diseñada con una arquitectura de múltiples páginas, permitiendo al usuario navegar a través de diferentes análisis de manera intuitiva.

Demostración en Vivo
Puedes explorar la aplicación en funcionamiento a través del siguiente enlace:

📊 Demostración en Vivo - stats-house-prices-edwinflores19.streamlit.app

Estructura del Proyecto y Funcionalidades
El proyecto está organizado en una página principal y varias sub-páginas, cada una dedicada a un análisis específico.

🏠 Página Principal
Proporciona una introducción al proyecto, una vista previa del conjunto de datos original y procesado, y un resumen estadístico básico. Es el punto de partida para entender los datos.

pages/2_📊_Análisis_Exploratorio.py
Esta página se centra en la visualización de datos (EDA - Exploratory Data Analysis). Contiene:

Un histograma para mostrar la distribución de precios.

Un diagrama de dispersión interactivo de Área vs. Precio.

Un mapa de calor de correlaciones para entender las relaciones entre las variables.

pages/3_📈_Predicción_de_Precio.py
Permite a los usuarios predecir el precio de una vivienda basándose en un modelo de Regresión Lineal. La página incluye un formulario interactivo en la barra lateral para que el usuario ingrese las características de una casa y obtenga una estimación de precio en tiempo real.

pages/4_🧩_Segmentación_de_Viviendas.py
Utiliza el algoritmo de Clustering K-Means para segmentar las viviendas en grupos naturales basados en su área y precio. La página visualiza los clústeres y proporciona un análisis de las características promedio de cada segmento.

pages/5_🕵️_Detección_de_Outliers.py
Una página dedicada a la detección de anomalías usando el modelo Isolation Forest. Identifica y visualiza las viviendas que son significativamente atípicas en el conjunto de datos, lo que es útil para el control de calidad de los datos o para encontrar propiedades únicas.

pages/6_🔑_Análisis_de_Características.py
Emplea un modelo de Random Forest para determinar y visualizar la importancia de las características. El usuario puede ver qué variables (como el área o el número de baños) tienen el mayor impacto en el precio de la vivienda.

pages/7_✨_Clasificación_de_Precio.py
Cambia el problema de regresión a un problema de clasificación. Esta página utiliza un modelo de K-Nearest Neighbors (KNN) para categorizar las viviendas en niveles de precio ('Bajo', 'Medio', 'Alto') y muestra la precisión del modelo.

pages/8_📊_Comparación_de_Modelos.py
Compara el rendimiento de diferentes modelos de Machine Learning (Regresión Lineal, Random Forest y Gradient Boosting) en la tarea de predicción de precios. Muestra métricas clave como el R² Score, MAE y MSE para ayudar a entender qué modelo es el más adecuado.

pages/9_🔮_Simulación_de_Escenarios.py
Un simulador interactivo que permite al usuario ajustar una sola característica de la vivienda (ej. el área) y ver cómo ese cambio afecta el precio predicho. Esta herramienta es ideal para la exploración de escenarios "what-if" y la toma de decisiones.

Cómo Ejecutar el Proyecto Localmente
Para clonar este repositorio y ejecutar la aplicación en tu máquina local, sigue estos pasos:

Clonar el repositorio:

git clone https://github.com/EdwinFlores19/Analisis-Estadistico-House-Prices-Streamlit.git
cd Analisis-Estadistico-House-Prices-Streamlit

Crear un entorno virtual (opcional pero recomendado):

python -m venv venv
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate

Instalar las dependencias:

pip install -r requirements.txt

Ejecutar la aplicación Streamlit:

streamlit run Pagina_Principal.py

La aplicación se abrirá automáticamente en tu navegador por defecto.

Dependencias
Este proyecto requiere las siguientes librerías de Python. Todas están listadas en el archivo requirements.txt.

streamlit

pandas

numpy

scikit-learn

plotly

Autor
Este proyecto fue desarrollado por Edwin Flores.
