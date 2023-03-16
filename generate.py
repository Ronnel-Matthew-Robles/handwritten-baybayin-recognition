from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import random

character = 'HA'
# input_folder = f'./cropped/{character}'
input_folder = f'.'

# filename = f'{input_folder}/{character}-{random.randint(0,2652)}.png'
filename = f'{input_folder}/dim.png'
print(filename)
image = Image.open(filename)

def show(img, figsize=(8, 4), title=None):
	print(f'Image details')
	print(f'Format {image.format}')
	print(f'Size {image.size}')
	print(f'Mode {image.mode}')
	plt.figure(figsize=figsize)
	plt.imshow(img, cmap="Greys")
	if title:
		plt.title(title)
	plt.show()

# show(image)
bw_image = image.convert(mode='L') #L is 8-bit black-and-white image mode
bw_image = ImageEnhance.Contrast(bw_image).enhance(3)
# show(bw_image)

# image = image.convert(mode='L')

# Invert sample, get bbox and display all that stuff.
inv_sample = ImageOps.invert(bw_image)
bbox = inv_sample.getbbox()

crop = inv_sample.crop(bbox)
# show(crop, title='Image cropped to bounding box')

# Helper function to get random diacritic based on name of d
def get_random_diacritic(d):
	diacritics_path = 'archive/Baybayin Diacritics'
	diacritic_path = f'{diacritics_path}/{d}'
	random_diacritic = f'{diacritic_path}/{d}-{random.randint(1,500)}.png'
	diacritic = Image.open(random_diacritic)
	bw_diacritic = diacritic.convert(mode='L')
	inv_diacritic = ImageOps.invert(bw_diacritic)
	return (inv_diacritic, random_diacritic)

#resize back
NEW_SIZE = 80

def add_diacritic_below(crop, diacritic_name):

	delta_w = NEW_SIZE - crop.size[0]
	delta_h = NEW_SIZE - crop.size[1]

	diacritic_max_size = delta_h-(delta_h//2)

	print(crop.size)
	print(diacritic_max_size)

	r, rname = get_random_diacritic(diacritic_name)
	if diacritic_name == 'cross':
		new_diacritic_size = random.randint(7,diacritic_max_size)
	else:
		new_diacritic_size = random.randint(5,diacritic_max_size)
	r = r.resize((new_diacritic_size,new_diacritic_size)) # ranges from 5-8
	# show(r, title=rname)
	print(new_diacritic_size)
	# Left Top Right Bottom
	padding = (delta_w//2, delta_h//2-new_diacritic_size, delta_w-(delta_w//2), delta_h-(delta_h//2)+new_diacritic_size)
	new_im = ImageOps.expand(crop, padding)

	dia_x = (NEW_SIZE//2)-(new_diacritic_size//2)
	dia_y = NEW_SIZE-delta_h-(delta_h//2)+new_diacritic_size+2
	to_add_x = random.randint(-8,14)
	# to_add_y = random.randint(0,10)
	new_im.paste(r, (dia_x+to_add_x, dia_y))

	return new_im


def add_diacritic_above(crop, diacritic_name):

	delta_w = NEW_SIZE - crop.size[0]
	delta_h = NEW_SIZE - crop.size[1]

	diacritic_max_size = delta_h-(delta_h//2)

	r, rname = get_random_diacritic(diacritic_name)
	new_diacritic_size = random.randint(5,diacritic_max_size)
	r = r.resize((new_diacritic_size,new_diacritic_size)) # ranges from 5-8
	# show(r, title=rname)

	# Left Top Right Bottom
	padding = (delta_w//2, delta_h//2+new_diacritic_size, delta_w-(delta_w//2), delta_h-(delta_h//2)-new_diacritic_size)
	new_im = ImageOps.expand(crop, padding)
	dia_x = (NEW_SIZE//2)-(new_diacritic_size//2)
	dia_y = delta_h//2
	to_add_x = random.randint(-8,8)
	# to_add_y = random.randint(0,10)
	new_im.paste(r, (dia_x+to_add_x, dia_y))

	return new_im

def enhance(img):
	na = np.array(img)

	# Check values
	# print(f'DEBUG: Min grey: {na.min()}, max grey: {na.max()}')

	# Get low (say 1%) and high (say 95%) percentiles
	loPct, hiPct = 1.0, 95.0
	loVal, hiVal = np.percentile(na, [loPct, hiPct])
	# print(f'DEBUG: {loPct} percentile={loVal}, {hiPct} percentile={hiVal}')
	
	# Scale image pixels from range loVal..hiVal to range 0..255
	res = ((na - na.min()) * 255.0 / (na.max() - na.min())).astype(np.uint8)

	img = Image.fromarray(res)

	denoised = img.filter(ImageFilter.MinFilter(1))
	bw_image = denoised.convert(mode="L")
	bw_image = ImageEnhance.Contrast(bw_image).enhance(4)
	bbox = bw_image.getbbox()
	crop = inv_sample.crop(bbox)

	return crop

if crop.size[1] >= 50:
	max_enhance = 3
	while crop.size[1] >= 50:
	    if (max_enhance > 0):
	        crop = enhance(crop)
	    else:
	        break
	    max_enhance -= 1
	    print(crop.size)

new_im_base = add_diacritic_below(crop, 'cross')
new_im_e = add_diacritic_below(crop, 'dot')
new_im_o = add_diacritic_above(crop, 'dot')

show(new_im_base, title=f'Resized and centered to {NEW_SIZE}x{NEW_SIZE} - base version')

show(new_im_e, title=f'Resized and centered to {NEW_SIZE}x{NEW_SIZE} - E/I version')

show(new_im_o, title=f'Resized and centered to {NEW_SIZE}x{NEW_SIZE} - O/U version')

# new_im_o.save('O_version.png')