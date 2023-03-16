
import time

# Import necessary libraries
import pandas as pd
import numpy as np

import cv2 as cv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays


# MODEL_NAME = f'3-conv-64-nodes-1-dense-512-dropout-0-{int(time.time())}'

# tensorboard = TensorBoard(log_dir=f'logs/{MODEL_NAME}')

xfile = 'baybayin_characters_x_test.npy'
yfile = 'baybayin_characters_y_test.npy'

# Load the numpy binary files
X = np.load(xfile)
y = np.load(yfile)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

x_train_normalized = tf.keras.utils.normalize(X_train, axis=1)  # scales data between 0 and 1
x_test_normalized = tf.keras.utils.normalize(X_test, axis=1)  # scales data between 0 and 1

IMG_SIZE=80
# -1 is a shorthand, which returns the length of the dataset
x_trainr = np.array(x_train_normalized).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_testr = np.array(x_test_normalized).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print("Training Samples dimension", x_trainr.shape)
print("Testing Samples dimension", x_testr.shape)

le = LabelEncoder()
y_train_final = le.fit_transform(y_train)
y_test_final = le.fit_transform(y_test)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


import os
from keras import models

directory = '/content/drive/MyDrive/Baybayin-Recognition/models'

for model_name in os.listdir(directory):
    model = models.load_model(model_name, custom_objects={'f1_m':f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
    print(f'evaluating {MODEL_NAME}')

    val_loss, val_acc, f1_score, precision, recall = model.evaluate(x_testr, y_test_final)  # evaluate the out of sample data with model
    print("Loss: ", val_loss)  # model's loss (error)
    print("Accuracy: ", val_acc)  # model's accuracy
    print("F1 measure: ", f1_score)  # model's f1_score
    print("Precision: ", precision)  # model's precision
    print("Recall: ", recall)  # model's recall
