# import module
from pdf2image import convert_from_path
import os

not_included = ['.DS_Store', os.path.basename(__file__), 'cropped']

# setting the path for scanned pdf and removing DS_store file from list
current_path = "."
current_list = [x for x in os.listdir(current_path) if x not in not_included]

for x in current_list:
	character_name = x.split('.')[0]
	print(character_name)

	# creates the directory if not yet present
	dir_path = f'cropped/{character_name}'
	os.makedirs(dir_path, exist_ok=True)

	# # # Store Pdf with convert_from_path function
	images = convert_from_path(x)

	for i in range(len(images)):

	# 	# Save pages as images in the pdf
		images[i].save(f'{dir_path}/{character_name}-{str(i+1)}.jpg', 'JPEG')
