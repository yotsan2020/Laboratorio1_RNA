import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Barra lateral personalizada
with st.sidebar:
    st.image("https://i.imgur.com/6MCLoH2.png", width=280)


# Cargar datos de MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28 * 28)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28 * 28)).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Cargar modelo preexistente
model = load_model('modelos_guardados/model_8.h5')

# Título
st.title("'evaluate' y 'predict' en TensorFlow/Keras")

# Crear listas de los argumentos para 'evaluate' y 'predict'
evaluate_args = {
    "x": "Datos de entrada",
    "y": "Datos de salida",
    "batch_size": "Tamaño del batch",
    "verbose": "Modo de verbosidad",
    "sample_weight": "Pesos de muestra (opcional)",
    "steps": "Número de pasos",
    "callbacks": "Lista de callbacks",
    "return_dict": "Si retorna dict o lista"
}

predict_args = {
    "x": "Datos de entrada",
    "batch_size": "Tamaño del batch",
    "verbose": "Modo de verbosidad",
    "steps": "Número de pasos",
    "callbacks": "Lista de callbacks"
}

# Crear las tablas en columnas
col1, col2 = st.columns(2)

with col1:
    st.subheader("Argumentos para 'evaluate'")
    st.write("Lista de argumentos para la función 'evaluate':")
    for arg, description in evaluate_args.items():
        st.write(f"- **{arg}**: {description}")

with col2:
    st.subheader("Argumentos para 'predict'")
    st.write("Lista de argumentos para la función 'predict':")
    for arg, description in predict_args.items():
        st.write(f"- **{arg}**: {description}")

# Ejemplo de uso con 'evaluate' en la columna 1
with col1:
    st.subheader("Ejemplo de uso con 'evaluate'")

    # Evaluar el modelo con los datos de test
    eval_results = model.evaluate(x_test, y_test, batch_size=32, verbose=1)

    # Mostrar resultados
    st.write("Resultados de la evaluación:")
    st.write(f"Pérdida: {eval_results[0]}")
    st.write(f"Exactitud: {eval_results[1]}")

# Ejemplo de uso con 'predict' en la columna 2
with col2:
    st.subheader("Ejemplo de uso con 'predict'")

    # Predecir con el modelo en los primeros 5 ejemplos del conjunto de test
    predictions = model.predict(x_test[:5], batch_size=32, verbose=1)

    # Mostrar resultados
    st.write("Predicciones para las primeras 5 imágenes del test:")
    st.write(predictions)
