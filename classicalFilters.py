import cv2
from scipy.signal import wiener
from PIL import Image, ImageFilter
import numpy as np

def gaussianBlur(image):
    return image.filter(ImageFilter.GaussianBlur(radius=2))

def medianFilter(image, kernel_size=3):
    image_cv = np.array(image)
    return Image.fromarray(cv2.medianBlur(image_cv, kernel_size))

def weinerFilter(image):
    image_np = np.array(image, dtype=float)
    filtered_image = wiener(image_np, (5, 5))
    filtered_image = np.nan_to_num(filtered_image, nan=0.0, posinf=255.0, neginf=0.0)
    return Image.fromarray(filtered_image.astype(np.uint8))

for i in range(10):
    imagePath = f'testImages/image_{i}.png'
    img = Image.open(imagePath)
    noisyImagePath = f'testImages/image_{i}_noisy.png'
    noisyImg = Image.open(noisyImagePath)
    
    gaussian_blurred = gaussianBlur(noisyImg)
    median_filtered = medianFilter(noisyImg)
    wiener_filtered = weinerFilter(noisyImg)

    combinedImages = [img, noisyImg, gaussian_blurred, median_filtered, wiener_filtered]
    combinedPath = f'testImages/image_{i}_classical.png'

    total_width = sum(image.width for image in combinedImages)
    max_height = max(image.height for image in combinedImages)
    collage = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for image in combinedImages:
        collage.paste(image, (x_offset, 0))
        x_offset += image.width
    collage.save(combinedPath)

