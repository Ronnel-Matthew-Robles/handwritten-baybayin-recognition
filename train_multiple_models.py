
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
    recall = K.clip(recall, 0, 1)  # ensure that the recall value is between 0 and 1
    return recall

def precision_m(y_true, y_pred):
    y_true = K.sigmoid(y_true)
    y_pred = K.sigmoid(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    f1 = 2*((precision*recall)/(precision+recall+K.epsilon()))
    f1 = K.clip(f1, 0, 1)  # ensure that the f1 value is between 0 and 1
    return f1

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128, 256, 512]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            MODEL_NAME = f'{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-80-dropout-{int(time.time())}'
            print(f'trainining {MODEL_NAME}')
            tensorboard = TensorBoard(log_dir=f'logs/{MODEL_NAME}')

            def construct_model(learningRate):
                smodel = Sequential()
                smodel.add(Conv2D(filters=layer_size, kernel_size=(3, 3), input_shape=x_trainr.shape[1:], activation='relu'))
                smodel.add(MaxPool2D((2, 2)))
                smodel.add(Dropout(0.8))
                
                for l in range(conv_layer-1):
                    smodel.add(Conv2D(filters=layer_size, kernel_size=(3, 3), activation='relu'))
                    smodel.add(MaxPool2D((2, 2)))
                    smodel.add(Dropout(0.8))

                smodel.add(Flatten())
                for l in range(dense_layer):
                    smodel.add(Dense(layer_size, activation='relu'))
                #     smodel.add(Dense(256, activation='relu'))

                smodel.add(Dense(63, activation='softmax'))
                optimizer = Adam(learning_rate=learningRate)
                smodel.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', f1_m,precision_m, recall_m])
            #     smodel.summary()
                return smodel

            model=construct_model(0.0001)

            model.fit(x_trainr, y_train_final, epochs=5, callbacks=[tensorboard], validation_data=(x_testr, y_test_final))  # train the model

            val_loss, val_acc, f1_score, precision, recall = model.evaluate(x_testr, y_test_final)  # evaluate the out of sample data with model
            print("Loss: ", val_loss)  # model's loss (error)
            print("Accuracy: ", val_acc)  # model's accuracy
            print("F1 measure: ", f1_score)  # model's f1_score
            print("Precision: ", precision)  # model's precision
            print("Recall: ", recall)  # model's recall

            model.save(f'models/{MODEL_NAME}')
