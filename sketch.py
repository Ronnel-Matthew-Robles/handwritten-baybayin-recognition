import gradio as gr
from gradio.templates import Paint
from keras import models
from PIL import Image
import numpy as np

labels = ['A',
 'B',
 'BA',
 'BE/BI',
 'BO/BU',
 'D',
 'DA',
 'DE/DI',
 'DO/DU',
 'E/I',
 'G',
 'GA',
 'GE/GI',
 'GO/GU',
 'H',
 'HA',
 'HE/HI',
 'HO/HU',
 'K',
 'KA',
 'KE/KI',
 'KO/KU',
 'L',
 'LA',
 'LE/LI',
 'LO/LU',
 'M',
 'MA',
 'ME/MI',
 'MO/MU',
 'N',
 'NA',
 'NE/NI',
 'NG',
 'NGA',
 'NGE/NGI',
 'NGO/NGU',
 'NO/NU',
 'O/U',
 'P',
 'PA',
 'PE/PI',
 'PO/PU',
 'R',
 'RA',
 'RE/RI',
 'RO/RU',
 'S',
 'SA',
 'SE/SI',
 'SO/SU',
 'T',
 'TA',
 'TE/TI',
 'TO/TU',
 'W',
 'WA',
 'WE/WI',
 'WO/WU',
 'Y',
 'YA',
 'YE/YI',
 'YO/YU']

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

model_name = 'models/3-conv-512-nodes-2-dense-80-dropout-1677260254'
# model_name = 'Experimentation/new_models/80-20-split/3-conv-512-nodes-2-dense-20+50-dropout-1678583391'
model = models.load_model(model_name, custom_objects={'f1_m':f1_m, 'precision_m': precision_m, 'recall_m': recall_m})

def classify_image(inp):
    i = Image.fromarray(inp, mode="RGB")
    im = i.convert('L')
    im = im.resize((80,80))
    img_array = np.array(im)
    # img_array = np.invert(img_array)
    # print(img_array)
    img_array = img_array / 255
    # plt.imshow(img_array)
    img_array = img_array.reshape((-1, 80, 80, 1))
    prediction = model.predict(img_array)[0]
    confidences = {labels[i]: float(prediction[i]) for i in range(63)}
    return confidences

# Define custom CSS styling
custom_css = """
.gradio-sketchpad {
    background-color: #F9F9F9;
}

/* Change border color of Sketchpad */
.gradio-sketchpad-canvas {
    border-color: #000000;
}
"""
paint_area = Paint(shape=[120,120])
interface = gr.Interface(fn=classify_image, inputs=paint_area, outputs=gr.Label(num_top_classes=3), css=custom_css)

# Launch the interface
interface.launch()
