import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from keras import models
from tensorflow.keras import backend as K

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
# model_name = '3-conv-64-nodes-1-dense-512-1669611358'
model_name = '3-conv-512-nodes-1-dense-80-dropout-1677258335'
model = models.load_model(model_name, custom_objects={'f1_m':f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
# model = models.load_model(model_name)
video = cv2.VideoCapture(1)
threshold = 140
# video.set(cv2.CAP_PROP_EXPOSURE,-4)

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

def get_character(image, contrast=3):
    denoised = image.filter(ImageFilter.MedianFilter())
    
    # 1. Convert grayscale to B&W and increase contrast
    bw_image = denoised.convert(mode='L') #L is 8-bit black-and-white image mode
    bw_image = ImageEnhance.Contrast(bw_image).enhance(contrast)

    bw_image = bw_image.point( lambda p: 255 if p > threshold else 0 )
    
    return bw_image

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
    crop = get_character(img, 4)
    return crop

def classify_image(frame, x1, y1, x2, y2):
    # Convert the captured frame into RGB
    img_from_array = Image.fromarray(frame.copy(), mode="RGB")

    # Crop part to be classified
    i = img_from_array.crop((x1,y1,x2,y2))

    # Get binarized image
    bw_image = get_character(i)

    # Resizing into 80x80 because we trained the model with this image size.
    im = bw_image.resize((80,80))

    # Convert the image to np array and inverting as we trained the model on inverted b&w
    img_array = np.array(im)
    img_array = np.invert(img_array)

    # Normalizing the values to 0-1
    img_array = img_array / 255

    # #Our keras model used a 4D tensor, (images x height x width x channel)
    # #So changing dimension 128x128x3 into 1x128x128x3 
    img_array_expanded = np.expand_dims(img_array, axis=0)

    # Get probabilities
    predictions = model.predict(img_array_expanded)

    # Get most likely character
    prediction = labels[np.argmax(predictions)]
    percentage = predictions[0][np.argmax(predictions)]
    return img_array, prediction, percentage


def draw_rectangle(img, label, x1, y1, x2, y2):
    # For bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (36,255,12), 2)

    # For the text background
    # Finds space required by the text so that we can put a background with that amount of width.
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    # Prints the text.    
    img = cv2.rectangle(img, (x1, y1 - 30), (x1 + w, y1), (36,255,12), -1)
    img = cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)


while True:
        _, frame = video.read()

        img_array, prediction, percentage = classify_image(frame, 110, 110, 410, 410)

        print(prediction)
        print(percentage)

        # cv2.rectangle(frame, (110, 110), (410, 410), (36,255,12), 2)
        draw_rectangle(frame, f'{prediction} {round(percentage*100, 2)}%', 110, 110, 410, 410)
        cv2.imshow("Capturing", frame)
        cv2.imshow("Binary and Resized", img_array)

        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()