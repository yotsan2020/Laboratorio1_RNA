import streamlit as st
import pandas as pd

# Barra lateral personalizada
with st.sidebar:
    st.image("https://i.imgur.com/6MCLoH2.png", width=280)



# Título e Introducción
st.title('Experimento: Flexibilidad Avanzada en la Definición de la Arquitectura de un Modelo de MMM')
st.markdown("""
Este experimento tiene como objetivo comparar tres enfoques para definir la arquitectura de un modelo de Machine Learning: **secuencial**, **funcional** y **subclassing**. En esta aplicación, se te permitirá ver las diferencias en la implementación y el impacto de añadir normalización de lotes (BatchNormalization) en cada enfoque.
""")

# Filtro para seleccionar el tipo de enfoque
option = st.selectbox(
    'Selecciona el enfoque de la arquitectura',
    ['Secuencial', 'Funcional', 'Subclassing']
)

# Disposición en columnas
col1, col2 = st.columns(2)

# Mostrar los códigos en columnas
with col1:
    st.subheader(f'{option} - Sin BatchNormalization')
    if option == 'Secuencial':
        # Aquí debes pegar tu código de la arquitectura Secuencial sin BatchNormalization
        st.code('''from tensorflow.keras import models, layers

# Construcción del modelo secuencial sin BatchNormalization
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
''')
    elif option == 'Funcional':
        # Aquí debes pegar tu código de la arquitectura Funcional sin BatchNormalization
        st.code('''from tensorflow.keras import layers, Model

# Entradas del modelo
inputs = layers.Input(shape=(28*28,))
x = layers.Dense(100, activation='relu')(inputs)

# Capas ocultas
for _ in range(9):
    x = layers.Dense(100, activation='relu')(x)

# Capa de salida
outputs = layers.Dense(10, activation='softmax')(x)

# Crear el modelo
model_functional = Model(inputs=inputs, outputs=outputs)

# Compilación del modelo
model_functional.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy', 'Recall', 'Precision', 'AUC'])
''')
    elif option == 'Subclassing':
        # Aquí debes pegar tu código de la arquitectura Subclassing sin BatchNormalization
        st.code('''import tensorflow as tf
from tensorflow.keras import layers

class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.dense_layers = []
        
        # Crear capas densas
        self.dense_layers.append(layers.Dense(100, activation='relu', input_shape=(28*28,)))
        for _ in range(9):
            self.dense_layers.append(layers.Dense(100, activation='relu'))
        
        # Capa de salida
        self.output_layer = layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return self.output_layer(x)

# Crear el modelo
model_subclassing = CustomModel()

# Compilación del modelo
model_subclassing.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy', 'Recall', 'Precision', 'AUC'])
''')

with col2:
    st.subheader(f'{option} - Con BatchNormalization')
    if option == 'Secuencial':
        # Aquí debes pegar tu código de la arquitectura Secuencial con BatchNormalization
        st.code('''from tensorflow.keras import models, layers

# Construcción del modelo secuencial con BatchNormalization
model_sequential_bn = models.Sequential()
model_sequential_bn.add(layers.Dense(100, activation='relu', input_shape=(28*28,)))
model_sequential_bn.add(layers.BatchNormalization())  # Añadir BatchNormalization

# Capas ocultas
for _ in range(9):
    model_sequential_bn.add(layers.Dense(100, activation='relu'))
    model_sequential_bn.add(layers.BatchNormalization())  # Añadir BatchNormalization

# Capa de salida
model_sequential_bn.add(layers.Dense(10, activation='softmax'))

# Compilación del modelo
model_sequential_bn.compile(optimizer='adam',
                             loss='categorical_crossentropy',
                             metrics=['accuracy', 'Recall', 'Precision', 'AUC'])
''')
    elif option == 'Funcional':
        # Aquí debes pegar tu código de la arquitectura Funcional con BatchNormalization
        st.code('''from tensorflow.keras import layers, Model

# Entradas del modelo
inputs = layers.Input(shape=(28*28,))
x = layers.Dense(100, activation='relu')(inputs)
x = layers.BatchNormalization()(x)  # Añadir BatchNormalization

# Capas ocultas
for _ in range(9):
    x = layers.Dense(100, activation='relu')(x)
    x = layers.BatchNormalization()(x)  # Añadir BatchNormalization

# Capa de salida
outputs = layers.Dense(10, activation='softmax')(x)

# Crear el modelo
model_functional_bn = Model(inputs=inputs, outputs=outputs)

# Compilación del modelo
model_functional_bn.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy', 'Recall', 'Precision', 'AUC'])
''')
    elif option == 'Subclassing':
        # Aquí debes pegar tu código de la arquitectura Subclassing con BatchNormalization
        st.code('''import tensorflow as tf
from tensorflow.keras import layers

class CustomModelWithBN(tf.keras.Model):
    def __init__(self):
        super(CustomModelWithBN, self).__init__()
        self.dense_layers = []
        
        # Crear capas densas
        self.dense_layers.append(layers.Dense(100, activation='relu', input_shape=(28*28,)))
        self.dense_layers.append(layers.BatchNormalization())  # Añadir BatchNormalization
        
        for _ in range(9):
            self.dense_layers.append(layers.Dense(100, activation='relu'))
            self.dense_layers.append(layers.BatchNormalization())  # Añadir BatchNormalization
        
        # Capa de salida
        self.output_layer = layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return self.output_layer(x)

# Crear el modelo
model_subclassing_bn = CustomModelWithBN()

# Compilación del modelo
model_subclassing_bn.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy', 'Recall', 'Precision', 'AUC'])
''')

# Espacio para la visualización del modelo
st.subheader(f'Visualización del Modelo ({option})')

# Colocar las imágenes debajo de los códigos correspondientes
if option == 'Secuencial':
    col1, col2 = st.columns(2)
    with col1:
        # Imagen de Secuencial sin BatchNormalization
        st.image('img_estilos\model_sequential.png', caption='Modelo Secuencial Sin BatchNormalization', use_column_width=True)
    with col2:
        # Imagen de Secuencial con BatchNormalization
        st.image('img_estilos\model_sequential_bn.png', caption='Modelo Secuencial Con BatchNormalization', use_column_width=True)

elif option == 'Funcional':
    col1, col2 = st.columns(2)
    with col1:
        # Imagen de Funcional sin BatchNormalization
        st.image('img_estilos\model_functional.png', caption='Modelo Funcional Sin BatchNormalization', use_column_width=True)
    with col2:
        # Imagen de Funcional con BatchNormalization
        st.image('img_estilos\model_functional_bn.png', caption='Modelo Funcional Con BatchNormalization', use_column_width=True)

elif option == 'Subclassing':
    col1, col2 = st.columns(2)
    with col1:
        # Imagen de Subclassing sin BatchNormalization
        st.image('img_estilos\model_subclassing.png', caption='Modelo Subclassing Sin BatchNormalization', use_column_width=True)
    with col2:
        # Imagen de Subclassing con BatchNormalization
        st.image('img_estilos\model_subclassing_bn.png', caption='Modelo Subclassing Con BatchNormalization', use_column_width=True)

# Título de la aplicación
st.title("Comparación de Enfoques de Modelos en Keras")

# Tabla comparativa utilizando Markdown
st.markdown("""
| **Enfoque**    | **Claridad y Rapidez de Implementación**                                   | **Flexibilidad para Redes Personalizadas y Complejas**   |
|----------------|----------------------------------------------------------------------------|----------------------------------------------------------|
| **Secuencial** | Rápido y sencillo. Simplemente agregas capas secuenciales.                 | Limitado para redes con múltiples entradas/salidas.      |
| **Funcional**  | Más complejo que el secuencial. Permite conexiones no lineales.            | Más flexible. Bueno para redes complejas.               |
| **Subclassing**| Requiere más código, el más complejo de implementar.                       | Altamente flexible. Permite total personalización.       |
""")