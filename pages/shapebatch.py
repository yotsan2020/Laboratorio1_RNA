import streamlit as st
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.utils import to_categorical

# Barra lateral personalizada
with st.sidebar:
    st.image("https://i.imgur.com/6MCLoH2.png", width=280)


# Cargar datos de MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28 * 28)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28 * 28)).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Función para crear y compilar el modelo
def create_model():
    model_sequential = models.Sequential()
    model_sequential.add(layers.Dense(100, activation='relu', input_shape=(28*28,)))
    
    # Capas ocultas
    for _ in range(9):
        model_sequential.add(layers.Dense(100, activation='relu'))
    
    # Capa de salida
    model_sequential.add(layers.Dense(10, activation='softmax'))
    
    # Compilación del modelo
    model_sequential.compile(optimizer='adam',
                             loss='categorical_crossentropy',
                             metrics=['accuracy', 'Recall', 'Precision', 'AUC'])
    
    return model_sequential

# Función para entrenar el modelo con un batch_size específico
def train_model(batch_size):
    # Crear y entrenar el modelo
    model = create_model()
    
    model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_data=(x_test, y_test), verbose=0)
    
    # Predicciones
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    
    y_pred_classes_train = np.argmax(y_pred_train, axis=1)
    y_pred_classes_test = np.argmax(y_pred_test, axis=1)
    
    # Reportes de clasificación
    target_names = [f"Clase {i}" for i in range(10)]
    train_report = classification_report(np.argmax(y_train, axis=1), y_pred_classes_train, target_names=target_names, digits=4)
    test_report = classification_report(np.argmax(y_test, axis=1), y_pred_classes_test, target_names=target_names, digits=4)
    
    # Matriz de confusión
    conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes_test)
    colorscale = [[0, "#EFEFEF"], [1, "#CC3F0C"]]
    fig = go.Figure(data=go.Heatmap(z=conf_matrix, colorscale=colorscale, x=target_names, y=target_names, showscale=False))
    
    # Añadir los valores en cada celda
    for i in range(len(target_names)):
        for j in range(len(target_names)):
            fig.add_annotation(text=str(conf_matrix[i, j]), x=target_names[j], y=target_names[i], showarrow=False, font=dict(color="black"))
    
    fig.update_layout(title="Matriz de Confusión", xaxis_title="Etiquetas Predichas", yaxis_title="Etiquetas Reales")
    
    return train_report, test_report, fig

# Crear el layout de Streamlit
st.title("Entrenamiento de Modelo MNIST")
st.write("Ajusta el tamaño del lote y observa cómo cambian las métricas del modelo.")

# Seleccionar tamaño del lote
batch_size = st.slider("Seleccionar tamaño de lote", min_value=16, max_value=10000, step=16, value=32)

# Entrenar el modelo con el tamaño de lote seleccionado
train_report, test_report, conf_matrix_fig = train_model(batch_size)

# Mostrar los resultados
col1, col2 = st.columns(2)

with col1:
    st.write("### Train Report")
    st.text(train_report)

with col2:
    st.write("### Test Report")
    st.text(test_report)

# Mostrar matriz de confusión
st.subheader("Matriz de Confusión")
st.plotly_chart(conf_matrix_fig)
