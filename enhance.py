from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np
from matplotlib import pyplot as plt


# images = [f'HA-{x}.png' for x in range(2669)]

# for img in images:

img = Image.open('HA-813.png')
na = np.array(img)
# Get low (say 1%) and high (say 95%) percentiles
loPct, hiPct = 1.0, 95.0
loVal, hiVal = np.percentile(na, [loPct, hiPct])

# Scale image pixels from range loVal..hiVal to range 0..255
res = ((na - na.min()) * 255.0 / (na.max() - na.min())).astype(np.uint8)

img = Image.fromarray(res)
denoised = img.filter(ImageFilter.MinFilter(5))
bw_image = denoised.convert(mode="L")
bw_image = ImageEnhance.Contrast(bw_image).enhance(2)

# Show image
plt.imshow(bw_image, cmap='gray'), plt.axis("off")
plt.show()