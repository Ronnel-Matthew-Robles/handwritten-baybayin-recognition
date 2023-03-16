import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from keras import models
from tensorflow.keras import backend as K
import imutils
from imutils.contours import sort_contours

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
model_name = 'models/3-conv-512-nodes-2-dense-80-dropout-1677260254'
model = models.load_model(model_name, custom_objects={'f1_m':f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
# model = models.load_model(model_name)
# video = cv2.VideoCapture(2)
threshold = 150
# video.set(cv2.CAP_PROP_EXPOSURE,-4)

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

def classify(video):
    while True:
            _, frame = video.read()

            # load the input image from disk, convert it to grayscale, and blur
            # it to reduce noise
            image = frame

            # Define the coordinates of the ROI
            x, y, w, h = 0, 0, 900, 700
            roi = frame[y:y+h, x:x+w]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # perform edge detection, find contours in the edge map, and sort the
            # resulting contours from left-to-right
            edged = cv2.Canny(blurred, 30, 150)
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            if (len(cnts) > 0):
                cnts = sort_contours(cnts, method="left-to-right")[0]
                # initialize the list of contour bounding boxes and associated
                # characters that we'll be OCR'ing
                chars = []

                # loop over the contours
                for c in cnts:
                    # compute the bounding box of the contour
                    (x, y, w, h) = cv2.boundingRect(c)
                    # filter out bounding boxes, ensuring they are neither too small
                    # nor too large
                    if (w >= 50 and w <= 195) and (h >= 60 and h <= 185):
                        # extract the character and threshold it to make the character
                        # appear as *white* (foreground) on a *black* background, then
                        # grab the width and height of the thresholded image
                        roi = gray[y:y + h, x:x + w]
                        thresh = cv2.threshold(roi, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                        (tH, tW) = thresh.shape
                        # if the width is greater than the height, resize along the
                        # width dimension
                        if tW > tH:
                            thresh = imutils.resize(thresh, width=100)
                        # otherwise, resize along the height
                        else:
                            thresh = imutils.resize(thresh, height=100)

                        # re-grab the image dimensions (now that its been resized)
                        # and then determine how much we need to pad the width and
                        # height such that our image will be 80x80
                        (tH, tW) = thresh.shape
                        dX = int(max(0, 200 - tW) / 2.0)
                        dY = int(max(0, 200 - tH) / 2.0)
                        # pad the image and force 32x32 dimensions
                        padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                            left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                            value=(0, 0, 0))
                        padded = cv2.resize(padded, (80, 80))
                        # cv2.imshow(str(x), padded)
                        # prepare the padded image for classification via our
                        # handwriting OCR model
                        padded = padded.astype("float32") / 255.0
                        padded = np.expand_dims(padded, axis=-1)
                        # update our list of characters that will be OCR'd
                        chars.append((padded, (x, y, w, h)))

                # extract the bounding box locations and padded characters
                boxes = [b[1] for b in chars]
                chars = np.array([c[0] for c in chars], dtype="float32")
                # OCR the characters using our handwriting recognition model
                preds = model.predict(chars)

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

            key=cv2.waitKey(1)
            if key == ord('q'):
                    break
    video.release()
    cv2.destroyAllWindows()


def update_output(cam_id):
    cap = cv2.VideoCapture(int(cam_id))
    classify(cap)

def select_camera():
    cam_id = int(camera_select.get())
    update_output(cam_id)

root = tk.Tk()
root.title("Camera Selector")

camera_select = ttk.Combobox(root, values=[0, 1, 2, 3])
camera_select.current(0)
camera_select.pack()

select_button = ttk.Button(root, text="Select Camera", command=select_camera)
select_button.pack()

root.mainloop()


def main():
    root = tk.Tk()
    root.title("Camera Selector")

    cameras = get_cameras()
    camera_var = tk.StringVar()
    camera_var.set(cameras[0])
    camera_dropdown = ttk.OptionMenu(root, camera_var, *cameras, command=lambda _: show_camera(cameras.index(camera_var.get())))
    camera_dropdown.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
