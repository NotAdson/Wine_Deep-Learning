from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt

# YOUR CODE HERE
shape = [8]
model = keras.Sequential([
    layers.Dense(units=512, activation="relu", input_shape=shape),
    layers.Dense(units=512, activation="relu"),
    layers.Dense(units=512, activation="relu"),
    layers.Dense(units=1)
])

activation_layer = layers.Activation('relu')

x = tf.linspace(-9, 9, 1000)
y = activation_layer(x) # once created, a layer is callable just like a function

plt.figure(dpi=100)
plt.plot(x, y)
plt.xlim(-3, 3)
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()