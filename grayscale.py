from PIL import Image
files = [f'WA-{x}.png' for x in range(2652, 2662)]
# files = ['TA-2666.png']
print(files)
for i in range(len(files)):
	img = Image.open(files[i]).convert('L')
	img.save(f'{files[i]}-new.png')