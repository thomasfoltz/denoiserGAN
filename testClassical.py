import cv2
import numpy as np
import torch, torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch.utils.data import Dataset
from model import Discriminator
from scipy.signal import wiener
from PIL import Image, ImageFilter

def gaussianNoise(image, mean=0., std=0.2):
    noise = torch.randn(image.size()) * std + mean
    noisyImage = image + noise
    return torch.clamp(noisyImage, 0., 1.)

class NoisyEMNIST(Dataset):
    def __init__(self, dataset, noiseFunction):
        self.dataset = dataset
        self.noiseFunction = noiseFunction
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        noisy_image = self.noiseFunction(image)
        return noisy_image, image

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

testDataEMNIST = torchvision.datasets.EMNIST(root='./data', split='balanced', train=False, download=False, transform=transform)
noisyTestDataEMNIST = NoisyEMNIST(testDataEMNIST, gaussianNoise)

# avgGaus, avgMed, avgWein = 0, 0, 0
# toPIL = ToPILImage()

for i in range(20):
    imagePath = f'results/image_{i+1}.png'
    img = Image.open(imagePath)
    noisyImagePath = f'results/image_{i+1}_noisy.png'
    noisyImg = Image.open(noisyImagePath)

    # noisyImg = toPIL(noisyTestDataEMNIST[i][0])
    
    gaussian_blurred = gaussianBlur(noisyImg)
    median_filtered = medianFilter(noisyImg)
    wiener_filtered = weinerFilter(noisyImg)

    print(f'image_{i+1} gaussian-blur conf: {round(discriminator(transform(gaussian_blurred).unsqueeze(0)).item()*100, 2)}%')
    print(f'image_{i+1} median-filter conf: {round(discriminator(transform(median_filtered).unsqueeze(0)).item()*100, 2)}%')
    print(f'image_{i+1} wiener-filter conf: {round(discriminator(transform(wiener_filtered).unsqueeze(0)).item()*100, 2)}%')

    # avgGaus+=discriminator(transform(gaussian_blurred).unsqueeze(0)).item()
    # avgMed+=discriminator(transform(median_filtered).unsqueeze(0)).item()
    # avgWein+=discriminator(transform(wiener_filtered).unsqueeze(0)).item()

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

# print(f'Average Confidence in Gaussian Blur:{round(avgGaus/18800*100, 2)}%')
# print(f'Average Confidence in Median Filtering:{round(avgMed/18800*100, 2)}%')
# print(f'Average Confidence in Weiner Filtering:{round(avgWein/18800*100, 2)}%')