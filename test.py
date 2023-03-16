# from PIL import Image, ImageEnhance, ImageOps
# import matplotlib.patches as patches
# import matplotlib.pyplot as plt
# import numpy as np

# input_folder = './cropped/PA/'

# filename = input_folder+'PA-13.png'
# image = Image.open(filename)

# def show(img, figsize=(8, 4), title=None):
# 	print(f'Image details')
# 	print(f'Format {image.format}')
# 	print(f'Size {image.size}')
# 	print(f'Mode {image.mode}')
# 	plt.figure(figsize=figsize)
# 	plt.imshow(img)
# 	if title:
# 		plt.title(title)
# 	plt.show()

# show(image)
# bw_image = image.convert(mode='L') #L is 8-bit black-and-white image mode
# bw_image = ImageEnhance.Contrast(bw_image).enhance(3)
# show(bw_image)

# # image = image.convert(mode='L')

# # Invert sample, get bbox and display all that stuff.
# inv_sample = ImageOps.invert(bw_image)
# bbox = inv_sample.getbbox()

# print(bbox)

# fig = plt.figure(figsize=(2, 2))
# ax = fig.add_axes([0,0,1,1])

# ax.imshow(inv_sample)
# rect = patches.Rectangle(
#     (bbox[0], bbox[3]), bbox[2]-bbox[0], -bbox[3]+bbox[1]-1,
#     fill=False, alpha=1, edgecolor='w')
# ax.add_patch(rect)
# # plt.show()

# crop = inv_sample.crop(bbox)
# show(crop, title='Image cropped to bounding box')

# #resize back
# new_size = 75
# delta_w = new_size - crop.size[0]
# delta_h = new_size - crop.size[1]
# padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
# new_im = ImageOps.expand(crop, padding)
# show(new_im, title='Resized and centered to 75x75')

import cv2
import numpy as np

# Load the image
img = cv2.imread("sample6.jpg")

# Define the coordinates of the ROI
x1, y1, x2, y2 = 10, 10, 100, 100

# Create the mask
mask = np.zeros_like(img)
mask[y1:y2, x1:x2] = 255

# Apply the mask
roi = cv2.bitwise_and(img, mask)

# Show the result
cv2.imshow("ROI", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
