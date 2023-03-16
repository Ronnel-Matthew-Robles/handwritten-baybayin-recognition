# import the necessary packages
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import argparse
import imutils
import cv2
from tensorflow.keras import backend as K
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import matplotlib.pyplot as plt

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

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
        help="path to input image")
ap.add_argument("-m", "--model", type=str, required=True,
        help="path to trained handwriting recognition model")
args = vars(ap.parse_args())

# load the handwriting OCR model
print("[INFO] loading handwriting OCR model...")
model = load_model(args["model"], custom_objects={'f1_m':f1_m, 'precision_m': precision_m, 'recall_m': recall_m})

# Constants
NEW_SIZE = 80

def show(img, figsize=(8, 4), title=None):
    print(f'Image details')
    # print(f'Format {image.format}')
    # print(f'Size {image.size}')
    # print(f'Mode {image.mode}')
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='Greys')
    if title:
        plt.title(title)
    plt.show()

# Helper function to resize and center an image
def resize_and_center(sample):
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

# load the input image from disk, convert it to grayscale, and blur
# it to reduce noise
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# cv2.imshow("Blurred", blurred)
# perform edge detection, find contours in the edge map, and sort the
# resulting contours from left-to-right
edged = cv2.Canny(gray, 10, 250)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]
# initialize the list of contour bounding boxes and associated
# characters that we'll be OCR'ing
chars = []

# cv2.imshow("Image", image)
# cv2.imshow("Gray", gray)
# cv2.imshow("Blurred", blurred)
# cv2.imshow("Edged", edged)
# cv2.waitKey(0)

# for c in cnts:
#     (x, y, w, h) = cv2.boundingRect(c)
#     if (w > 80 and w < 200) and (h > 80 and h < 200):
#         roi = gray[y+12:y + h-12, x+12:x + w-12]
#         thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#         # cv2.imshow(str(x), thresh)
#         (tH, tW) = thresh.shape
#         if tW > tH:
#             thresh = imutils.resize(thresh, width=100)
#         else:
#             thresh = imutils.resize(thresh, height=100)
#         (tH, tW) = thresh.shape
#         dX = int(max(0, 200 - tW) / 2.0)
#         dY = int(max(0, 200 - tH) / 2.0)
#         # pad the image and force 32x32 dimensions
#         padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
#         padded = cv2.resize(padded, (80, 80))
#         # cv2.imshow(str(x), padded)
#         padded = padded.astype("float32") / 255.0
#         padded = np.expand_dims(padded, axis=-1)
#         chars.append((padded, (x, y, w, h)))

# # # loop over the contours
# for c in cnts:
#         # compute the bounding box of the contour
#     (x, y, w, h) = cv2.boundingRect(c)
#     # filter out bounding boxes, ensuring they are neither too small
#     # nor too large
#     if (w > 80 and w < 200) and (h > 80 and h < 200):
#     # extract the character and threshold it to make the character
#     # appear as *white* (foreground) on a *black* background, then
#     # grab the width and height of the thresholded image
#     roi = gray[y:y + h, x:x + w]
#     thresh = cv2.threshold(roi, 0, 255,
#             cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#     (tH, tW) = thresh.shape
#     # if the width is greater than the height, resize along the
#     # width dimension
#     if tW > tH:
#         thresh = imutils.resize(thresh, width=100)
#     # otherwise, resize along the height
#     else:
#         thresh = imutils.resize(thresh, height=100)

#     # re-grab the image dimensions (now that its been resized)
#     # and then determine how much we need to pad the width and
#     # height such that our image will be 80x80
#     (tH, tW) = thresh.shape
#     dX = int(max(0, 200 - tW) / 2.0)
#     dY = int(max(0, 200 - tH) / 2.0)
#     # pad the image and force 32x32 dimensions
#     padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
#             left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
#             value=(0, 0, 0))
#     padded = cv2.resize(padded, (80, 80))
#     cv2.imshow(str(x), padded)
#     # prepare the padded image for classification via our
#     # handwriting OCR model
#     padded = padded.astype("float32") / 255.0
#     padded = np.expand_dims(padded, axis=-1)
#     # update our list of characters that will be OCR'd
#     chars.append((padded, (x, y, w, h)))

# loop over the contours
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    if (w > 50 and w < 180) and (h > 50 and h < 180):
        roi = gray[y:y + h, x:x + w]
        roi_image = Image.fromarray(roi)
        crop = get_character(roi_image)

        # if crop.size[1] >= 50:
        #     # Add limiter to avoid infinite loop
        #     max_enhance = 3
        #     while crop.size[1] >= 50:
        #         if (max_enhance > 0):
        #             crop = enhance(crop)
        #         else:
        #             break
        #         max_enhance -= 1

        new_im = resize_and_center(crop)
        # show(new_im)
        # new_im = new_im.resize((80,80))
        new_im_array = np.expand_dims(np.array(new_im), axis=-1)
        cv2.imshow(str(c), new_im_array)
        chars.append((new_im_array, (x, y, w, h)))

# extract the bounding box locations and padded characters
boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32") / 255.0
# chars = np.array([c[0] for c in chars], dtype="float32")
print(chars.shape)
# chars_reshaped = np.array(chars).reshape(-1, NEW_SIZE, NEW_SIZE, 1)

# OCR the characters using our handwriting recognition model
preds = model.predict(chars)
# define the list of label names
labelNames = ['A',
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

 # loop over the predictions and bounding box locations together
for (pred, (x, y, w, h)) in zip(preds, boxes):
    # find the index of the label with the largest corresponding
    # probability, then extract the probability and label
    i = np.argmax(pred)
    prob = pred[i]
    label = labelNames[i]
    # draw the prediction on the image
    print("[INFO] {} - {:.2f}%".format(label, prob * 100))
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    # show the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
