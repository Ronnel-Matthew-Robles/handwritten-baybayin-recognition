import cv2
import os
import glob

# setting the path for scanned images and removing DS_store file from list
scanned_path = "scanned"
scanned_list = [x for x in os.listdir(scanned_path) if x != '.DS_Store' and os.path.isfile(f'{scanned_path}/{x}')]

# scanning files inside folders if present
folders_scanned_list = glob.glob(f'{scanned_path}/*/*.jpg')
# print(folders_scanned_list)
new_folders_scanned_list = [x[8:] for x in folders_scanned_list]

# prints all files
print("Files and directories in '", scanned_path, "' :")
full_scanned_list = scanned_list + new_folders_scanned_list
print(full_scanned_list)

# used to keep track of total number of cropped items
total_cropped = 0

# automatically crops boxes from scanned files
for scan in full_scanned_list:
	# reading image file
	image = cv2.imread(f'scanned/{scan}')
	character_name = scan.split('/')[1].split('-')[0]
	print(f'Scanning {scan}')

	# creates the directory if not yet present
	dir_path = f'cropped/{character_name}'
	os.makedirs(dir_path, exist_ok=True)

	# START OF COUNTOURING
	# converting to gray scale
	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	# applying canny edge detection
	edged = cv2.Canny(gray, 10, 250)

	# finding contours
	(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#END OF COUNTOURING

	# getting current number of cropped images from cropped folder
	idx = 0
	# Iterate directory
	for path in os.listdir(dir_path):
	    # check if current path is a file
	    if os.path.isfile(os.path.join(dir_path, path)):
	        idx += 1
	idx -= 1

	matched_contours = []

	for c in cnts:
		x,y,w,h = cv2.boundingRect(c)
		if (w>80 and h>80) and (w<180 and h<180):
		# if (w>100 and h>100) and (w<120 and h<120):
		# if (w>80 and h>80) and (w<100 and h<100):
			matched_contours.append(c)
			idx+=1

			# crop image
			# new_img=image[y+12:y+h-12,x+12:x+w-12]
			new_img=gray[y+12:y+h-12,x+12:x+w-12]
			# cv2.imshow(str(idx), new_img)
			# new_img=image[y:y+h,x:x+w]

			# save crop image
			# cv2.imwrite(f'{dir_path}/{character_name}-{str(idx)}.png', new_img)
			
			total_cropped += 1

	print(len(matched_contours))
	# cv2.imshow("Original Image",image)
	# cv2.imshow("Grayscale", gray)
	# cv2.imshow("Canny Edge", edged)
	cv2.drawContours(image, matched_contours, -1, (0, 255, 0), 3)
	cv2.imshow('Contours', image)
	cv2.waitKey(0)

	print(f'File count in {character_name} folder:', idx)

print('>> ' + str(total_cropped) + ' Objects Cropped Successfully!')
print(">> Check out 'cropped' Directory")
