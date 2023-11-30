import cv2
from model import Discriminator
import numpy as np
from scipy.signal import wiener
from PIL import Image, ImageFilter
import torch
import torchvision.transforms as transforms

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

discriminator = Discriminator()
discriminator.load_state_dict(torch.load('./discriminator.pth'))
discriminator.eval()
transform = transforms.Compose([transforms.ToTensor()])

for i in range(20):
    imagePath = f'results/image_{i+1}.png'
    img = Image.open(imagePath)
    noisyImagePath = f'results/image_{i+1}_noisy.png'
    noisyImg = Image.open(noisyImagePath)
    
    gaussian_blurred = gaussianBlur(noisyImg)
    median_filtered = medianFilter(noisyImg)
    wiener_filtered = weinerFilter(noisyImg)

    print(f'image_{i+1} gaussian-blur conf: {round(discriminator(transform(gaussian_blurred).unsqueeze(0)).item()*100, 2)}%')
    print(f'image_{i+1} median-filter conf: {round(discriminator(transform(median_filtered).unsqueeze(0)).item()*100, 2)}%')
    print(f'image_{i+1} wiener-filter conf: {round(discriminator(transform(wiener_filtered).unsqueeze(0)).item()*100, 2)}%')
    print('\n')

    combinedImages = [img, noisyImg, gaussian_blurred, median_filtered, wiener_filtered]
    combinedPath = f'results/image_{i+1}_classical.png'

    total_width = sum(image.width for image in combinedImages)
    max_height = max(image.height for image in combinedImages)
    collage = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for image in combinedImages:
        collage.paste(image, (x_offset, 0))
        x_offset += image.width
    collage.save(combinedPath)

