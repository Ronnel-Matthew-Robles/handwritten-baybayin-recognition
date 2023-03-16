import numpy as np
from keras import models
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt

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
# one hot encode target values

classes = list(le.classes_)

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

#Load the saved model
model_name = '3-conv-512-nodes-1-dense-80-dropout-1677258335'
model = models.load_model(model_name, custom_objects={'f1_m':f1_m, 'precision_m': precision_m, 'recall_m': recall_m})

predictions = model.predict(x_testr)

y_predicted_labels = [np.argmax(i) for i in predictions]

cm = tf.math.confusion_matrix(labels=y_test_final, predictions=y_predicted_labels)

import seaborn as sn
plt.figure(figsize = (25,20))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

# Save the plot as a PNG image
plt.savefig('confusion_matrix.png')