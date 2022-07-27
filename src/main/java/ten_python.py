# TensorFlow and tf.keras
from pyexpat import model
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pprint
import os

print(os.getcwd())
print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0


try:
   model=tf.keras.models.load_model("./my_mod",compile=False)
except OSError:
   print("failed load model")
# model=tf.keras.Sequential([
#    tf.keras.layers.Flatten(input_shape=(28,28)),
#    tf.keras.layers.Dense(128, activation='relu'),
#    tf.keras.layers.Dense(10)
#    ])

# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# print(model)
# model.fit(train_images, train_labels, epochs=10)

# test_loss,test_acc=model.evaluate(test_images, test_labels, verbose=2)

# probability_model=tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# prediction=probability_model.predict(test_images)
prediction=model.predict(test_images)
print("output:"+str(prediction[0]))
print("input:"+str(test_images[0][0]))
print(np.argmax(prediction[0]))


# pprint.pprint(prediction)


# probability_model.save("saved_model/my_mod")