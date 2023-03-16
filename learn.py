# Import necessary libraries
import pandas as pd
import numpy as np                               # Import numpy
from skimage import data, io   # Import skimage library (data - Test images and example data.
#                          io - Reading, saving, and displaying images.)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt                  # Import matplotlib.pyplot (Plotting framework in Python.)
# %matplotlib inline
import os                                        # This module provides a portable way of using operating system dependent functionality.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
# Suppress warnings
warnings.filterwarnings('ignore')
from IPython.display import display
import cv2 as cv
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
import seaborn as sns
import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
from ann_visualizer.visualize import ann_viz
import visualkeras

mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
(x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

x_train = tf.keras.utils.normalize(x_train, axis=1)  # scales data between 0 and 1
x_test = tf.keras.utils.normalize(x_test, axis=1)  # scales data between 0 and 1

print(x_train.shape)
print(x_test.shape)

def construct_model(learningRate):
    smodel = Sequential()
    smodel.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(75, 75, 1), activation='relu'))
    smodel.add(MaxPool2D((2, 2)))
    smodel.add(Dropout(0.8))
    smodel.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    smodel.add(MaxPool2D((2, 2)))
    smodel.add(Dropout(0.8))
    smodel.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    smodel.add(MaxPool2D((2, 2)))
    smodel.add(Dropout(0.8))
    smodel.add(Flatten())
    smodel.add(Dense(256, activation='relu'))
    smodel.add(Dense(256, activation='relu'))
    smodel.add(Dense(63, activation='softmax'))
    optimizer = Adam(learning_rate=learningRate)
    smodel.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # smodel.summary()
    return smodel

# model=construct_model(0.0001)
# ann_viz(model, view=True, filename='construct_model', title='CNN - Model 1 — Simple Architecture')

# visualkeras.layered_view(model, legend=True, to_file='model.png').show()

# model.fit(x_train, y_train, epochs=50)  # train the model

# val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
# print("Loss: ", val_loss)  # model's loss (error)
# print("Accuracy: ", val_acc)  # model's accuracy

# model.save('baybayin_recognition')
# predictions = model.predict([x_test])

# tf.keras.utils.plot_model(
# model,
# to_file="model.png",
# show_shapes=True,
# show_dtype=False,
# show_layer_names=True,
# rankdir="TB",
# expand_nested=True,
# dpi=96,
# layer_range=None,
# show_layer_activations=True,
# )
# visualkeras.layered_view(model, legend=True)