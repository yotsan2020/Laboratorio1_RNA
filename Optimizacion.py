import streamlit as st
import pandas as pd
import pickle
import os
import plotly.graph_objects as go
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.utils import to_categorical

# Cargar resultados
df = pd.read_csv("model_results.csv")

# Extraer el f1-score macro avg del test report
def extract_f1_score(report):
    lines = report.split("\n")
    for line in lines:
        if "macro avg" in line:
            return float(line.split()[-2])
    return None

df["f1_macro_avg"] = df["test_report"].apply(extract_f1_score)


# --- Introducción ---
st.title("Optimización de Hiperparámetros para Máxima Precisión")

st.markdown("""
### Objetivo del experimento
Buscamos optimizar los hiperparámetros para maximizar la precisión en un modelo de clasificación de imágenes. 
Probamos combinaciones de:
- **Optimizador**: Adam, SGD
- **Batch Size**: 32, 64, 128
- **Épocas**: 10, 20
- **Capas**: 5,10 
- **Neuronas**: 50, 100
""")

# --- Tabla filtrable por arquitectura ---
st.subheader("Comparación de Modelos")

# Filtros
num_layers = st.multiselect("Selecciona el número de capas", sorted(df["num_layers"].unique()))
num_neurons = st.multiselect("Selecciona el número de neuronas", sorted(df["num_neurons"].unique()))

filtered_df = df.copy()
if num_layers:
    filtered_df = filtered_df[filtered_df["num_layers"].isin(num_layers)]
if num_neurons:
    filtered_df = filtered_df[filtered_df["num_neurons"].isin(num_neurons)]

# Mostrar tabla ordenada por f1-score macro avg
display_df = filtered_df[["model_name","optimizer","batch_size","epochs","num_layers","num_neurons","f1_macro_avg"]].sort_values(by="f1_macro_avg", ascending=False)
st.dataframe(display_df,use_container_width=True)

# --- Buscador de modelos ---
st.subheader("Detalles de Modelos")
selected_model = st.selectbox("Selecciona un modelo para ver sus detalles", df["model_name"].unique())

if selected_model:
    model_data = df[df["model_name"] == selected_model].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Train Report")
        st.text(model_data["train_report"])
    
    with col2:
        st.write("### Test Report")
        st.text(model_data["test_report"])

    # Mostrar matriz de confusión
    st.subheader("Matriz de Confusión")
    conf_matrix_path = model_data["conf_matrix_path"]

    if os.path.exists(conf_matrix_path):
        with open(conf_matrix_path, "rb") as f:
            fig = pickle.load(f)

        # Verificar si el objeto cargado es una figura de Plotly
        if isinstance(fig, go.Figure):
            st.plotly_chart(fig)
        else:
            st.error("El archivo no contiene una figura de Plotly válida.")
    else:
        st.error("No se encontró la matriz de confusión.")

    model_path = model_data["model_path"]
    model = tf.keras.models.load_model(model_path)

    # --- Subir imagen para predicción ---
    st.subheader("Predicción con Modelo Seleccionado")
    uploaded_file = st.file_uploader("Sube una imagen en escala de grises (28x28)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")  # Convertir a escala de grises
        image = image.resize((28, 28))  # Asegurar tamaño 28x28
        image_array = np.array(image).astype('float32') / 255  # Normalizar
        image_array = image_array.reshape((1, 28 * 28))  # Aplanar
        
        # Cargar modelo
        
        if os.path.exists(model_path):
            
            prediction = model.predict(image_array)
            predicted_class = np.argmax(prediction)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Imagen subida", use_column_width=True)
            
            with col2:
                st.write("### Predicción del Modelo")
                st.write(f"El modelo predice: **{predicted_class}**")
        else:
            st.error("No se encontró el modelo seleccionado.")


def plot_image_histogram(predictions, img_no):
    """
    Genera el histograma para una imagen específica basada solo en las predicciones.
    """
    # Obtener las predicciones para la imagen seleccionada
    prediction = predictions[img_no]
    
    # Crear la figura para el histograma de las predicciones
    fig = go.Figure()

    # Histograma de la predicción
    fig.add_trace(go.Bar(
        x=list(range(10)),
        y=prediction,
        name='Predicción',
        marker=dict(color='blue', line=dict(color='black', width=1)),
    ))

    # Actualizar diseño
    fig.update_layout(
        title=f"Distribución de Probabilidades para la Imagen {img_no}",
        xaxis_title="Clase",
        yaxis_title="Probabilidad",
        barmode='group',
        bargap=0.1,
        showlegend=False,
    )

    return fig

# Cargar datos de MNIST
(x_train, y_train_labels), (x_test, y_test_labels) = tf.keras.datasets.mnist.load_data()
x_test = x_test.reshape((x_test.shape[0], 28 * 28)).astype('float32') / 255
y_test = to_categorical(y_test_labels)

# Obtener las predicciones para el conjunto de prueba
predictions = model.predict(x_test)

# Identificar las imágenes mal clasificadas
incorrectly_classified = []
for i in range(len(x_test)):
    predicted_class = np.argmax(predictions[i])
    true_class = np.argmax(y_test[i])
    if predicted_class != true_class:
        incorrectly_classified.append({
            'Real': true_class,
            'Predicho': predicted_class,
            'Índice': i
        })

# Crear un DataFrame con las imágenes mal clasificadas
df_incorrect = pd.DataFrame(incorrectly_classified)

# 1. Crear las dos columnas en Streamlit
col1, col2 = st.columns(2)

# 2. Columna 1: Mostrar las imágenes mal clasificadas en una tabla
with col1:
    st.title("Imágenes Mal Clasificadas")
    st.write("Listado de imágenes mal clasificadas:")
    st.dataframe(df_incorrect)

# 3. Columna 2: Buscador para seleccionar el índice de la imagen
with col2:
    st.write("Selecciona el índice de una imagen mal clasificada para ver su histograma:")
    
    # Seleccionar índice de la imagen mal clasificada
    img_no = st.selectbox("Selecciona el índice", df_incorrect['Índice'])

    # Mostrar el histograma para la imagen seleccionada
    image_histogram = plot_image_histogram(predictions, img_no)
    st.plotly_chart(image_histogram)

    # Mostrar la imagen seleccionada
    st.write("Imagen seleccionada:")
    selected_image = x_test[img_no].reshape(28, 28)  # Reshape para mostrarla correctamente
    st.image(selected_image, caption=f"Índice: {img_no}", use_column_width=True)

from sklearn.metrics import precision_recall_curve, roc_curve

def plot_curves(predictions, y_true, class_no):
    """
    Genera las curvas Precision-Recall y ROC para una clase específica,
    resaltando el mejor umbral.
    """
    # Obtener las probabilidades de la clase seleccionada
    y_scores = predictions[:, class_no]
    y_labels = (y_true.argmax(axis=1) == class_no).astype(int)
    
    # Calcular curva Precision-Recall
    precision, recall, thresholds_pr = precision_recall_curve(y_labels, y_scores)
    distances_pr = np.sqrt((1 - recall) ** 2 + (1 - precision) ** 2)
    best_threshold_pr = thresholds_pr[np.argmin(distances_pr)]
    best_pr_point = (recall[np.argmin(distances_pr)], precision[np.argmin(distances_pr)])
    
    # Calcular curva ROC
    fpr, tpr, thresholds_roc = roc_curve(y_labels, y_scores)
    distances_roc = np.sqrt(fpr ** 2 + (1 - tpr) ** 2)
    best_threshold_roc = thresholds_roc[np.argmin(distances_roc)]
    best_roc_point = (fpr[np.argmin(distances_roc)], tpr[np.argmin(distances_roc)])
    
    # Crear figuras con Plotly
    fig = go.Figure()
    
    # Agregar la Curva Precision-Recall
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[best_pr_point[0]], y=[best_pr_point[1]], mode='markers', 
                             marker=dict(color='red', size=10), name=f'Mejor Umbral PR ({best_threshold_pr:.2f})'))
    
    fig.update_layout(title=f'Curva Precision-Recall para Clase {class_no}', xaxis_title='Recall', yaxis_title='Precision')
    
    # Mostrar gráfico PR
    st.plotly_chart(fig)
    
    # Crear figura para la Curva ROC
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC', line=dict(color='green')))
    fig_roc.add_trace(go.Scatter(x=[best_roc_point[0]], y=[best_roc_point[1]], mode='markers', 
                                 marker=dict(color='red', size=10), name=f'Mejor Umbral ROC ({best_threshold_roc:.2f})'))
    
    fig_roc.update_layout(title=f'Curva ROC para Clase {class_no}', xaxis_title='FPR (Tasa de Falsos Positivos)',
                          yaxis_title='TPR (Tasa de Verdaderos Positivos)')
    
    # Mostrar gráfico ROC
    st.plotly_chart(fig_roc)

# Crear la interfaz en Streamlit
st.title("Umbral Óptimo para Cada Clase")

# Selector de clase
class_no = st.selectbox("Selecciona una clase para analizar", range(10))

# Generar y mostrar curvas
plot_curves(predictions, y_test, class_no)
