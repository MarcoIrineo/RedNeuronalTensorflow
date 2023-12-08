from __future__ import absolute_import, division, print_function, unicode_literals

# se importa la biblioteca Tensorflow y Keras
import tensorflow as tf
from tensorflow import keras

# Se importa numpy para manejo de datos y matplotlib para grafica
import numpy as np
import matplotlib.pyplot as plt

#Se imprime la versión de tensorflow
print(tf.__version__)

#Importar el conjunto de datos Fashion MNIST
fashion_mnist = keras.datasets.fashion_mnist

#Cargar el conjunto de datos, 
#Las imagenes y etiquetas que se usarán para el entrenamiento estan en (train_images, train_labels)
#Las imagenes y etiquetas que se usarán para la prueba estan en ((test_images, test_labels)
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['Blusa/Camiseta', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
               'Sandalias', 'Camisa', 'Tenis', 'Bolsa', 'Bota']

#Obtener la forma del arreglo numpy con las imágenes de entrenamiento
train_images.shape

#Obtener la cantidad de elementos en el arreglo numpy con las etiquetas de entrenamiento 
len(train_labels)

#Mostrar el arreglo de etiquetas
train_labels

#Obtener la forma del arreglo numpy con las imágenes de prueba
test_images.shape

#Obtener la cantidad de elementos en el arreglo numpy con las etiquetas de prueba 
len(test_labels)

#Mostrar la primer imagen
plt.figure()
plt.imshow(train_images[0])
#Colocar una barra de colores que indica los valores de los pixeles
plt.colorbar()
plt.grid(False)
plt.show()

#Se divide el conjunto de entrenamiento sobre 255
train_images = train_images / 255.0
#Se divide el conjunto de prueba sobre 255
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    #Grafica en la subgráfica i 
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # Grafica la imagen i
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    #Incluye la etitqueta de la imagen 1
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#Se definen las capas
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Ajuste del modelo a los datos de entrenamiento
model.fit(train_images, train_labels, epochs=5)

#Evaluar el modelo en los datos de prueba, la pérdida (loss) se almacena en test_loss, la exactitud en test_acc
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

#Predecir sobre las imagenes de prueba
predictions = model.predict(test_images)

predictions[0]

#Encotrar la etiqueta con el valor más alto
np.argmax(predictions[0])

test_labels[0]

#Función para graficar la imagen predicha
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)
  
  predicted_label = np.argmax(predictions_array)
    #Si hay una discrepancia entre la clase predicha y la correcta entonces el color es rojo, si todo esta correcto es azúl.
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  #Coloca el texto con el nombre de la clase y la confianza de predicción
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
#Función para graficar el arreglo de predicciones
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
    #Si hay una discrepancia entre la clase predicha y la correcta entonces el color es rojo, si todo esta correcto es azúl.

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

  i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
#graficar la imagen predicha
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
#graficar el arreglo de predicciones
plot_value_array(i, predictions,  test_labels)
plt.show()


i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
#graficar la imagen predicha
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
#graficar el arreglo de predicciones
plot_value_array(i, predictions,  test_labels)
plt.show()

# Graficar las primeras X imagenes de prueba, su etiqueta predicha y la verdadera etiqueta
# Las predicciones correctas están en azúl, las incorrectas en rojo
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# Toma una imagen del conjunto de datos de prueba
img = test_images[0]

print(img.shape)

# Agrega la imagen a un lote en donde es la única imagen
img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)


#Grafica las predicciones
plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()

prediction_result = np.argmax(predictions_single[0])
print(prediction_result)