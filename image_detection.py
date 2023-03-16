import tensorflow as tf
from keras import models
from PIL import Image
import numpy as np
import gradio as gr 
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

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
model = models.load_model(model_name, custom_objects={'f1_m':f1_m, 'precision_m': precision_m, 'recall_m': recall_m})

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

NEW_SIZE = 80

# Helper function to resize and center an image
def resize_and_center(crop):
    delta_w = NEW_SIZE - crop.size[0]
    delta_h = NEW_SIZE - crop.size[1]
    # Left Top Right Bottom
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    return ImageOps.expand(crop, padding)

# Main function to get the character from image
def get_character(image, contrast=3, invert=True):
    denoised = image.filter(ImageFilter.MedianFilter())

    # 1. Convert grayscale to B&W and increase contrast
    bw_image = denoised.convert(mode='L') #L is 8-bit black-and-white image mode
    bw_image = ImageEnhance.Contrast(bw_image).enhance(contrast)

    if invert:
        # 2. Invert image, get bounding box, and display
        inv_sample = ImageOps.invert(bw_image)
        bbox = inv_sample.getbbox()

        # 3. Crop the character inside bounding box
        crop = inv_sample.crop(bbox)
    else:
        bbox = bw_image.getbbox()
        crop = bw_image.crop(bbox)

    return crop

# Converts the image to binary and removes background noise for better crop
def enhance(img):
    # Convert image to numpy array
    na = np.array(img)

    # Get low (say 1%) and high (say 95%) percentiles
    loPct, hiPct = 1.0, 95.0
    loVal, hiVal = np.percentile(na, [loPct, hiPct])

    # Scale image pixels from range loVal..hiVal to range 0..255
    res = ((na - na.min()) * 255.0 / (na.max() - na.min())).astype(np.uint8)

    # Convert numpy array to image
    img = Image.fromarray(res)

    # Get character from binary image
    crop = get_character(img, 4, False)
    return crop

def classify_image(inp):
    i = Image.fromarray(inp, mode="RGB")
    im = i.convert('L')
    crop = get_character(im)
    new_im = resize_and_center(crop)
    im = new_im.resize((80,80))
    img_array = np.array(im)
    # img_array = np.invert(img_array)
    img_array = img_array / 255
    print(img_array)
    img_array = img_array.reshape((-1, 80, 80, 1))
    prediction = model.predict(img_array)[0]
    confidences = {labels[i]: float(prediction[i]) for i in range(63)}
    return confidences

# filename = f'cropped/A/A-0.png'
# image = Image.open(filename)

# classify_image(image)

gr.Interface(fn=classify_image, inputs=gr.Image(shape=(80, 80)), outputs=gr.Label(num_top_classes=3)).launch()
# gr.Interface(fn=classify_drawing, inputs="sketchpad", outputs=gr.Label(num_top_classes=3),live=True).launch(share=True)
