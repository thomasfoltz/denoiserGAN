import numpy as np
import os
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

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

transform = transforms.Compose([transforms.ToTensor(),])
trainDataEMNIST = torchvision.datasets.EMNIST(root='./data', split='balanced', train=True, 
                                                download=False, transform=transform)

noisyTrainDataEMNIST = NoisyEMNIST(trainDataEMNIST, gaussianNoise)

if not os.path.exists('images/'): os.makedirs('images/')
toPIL = transforms.ToPILImage()
for i in range(10):
    img = toPIL(trainDataEMNIST.data[i])
    img.save(f'images/image_{i}.png')
    img = toPIL(noisyTrainDataEMNIST[i][0])
    img.save(f'images/image_{i}_noisy.png')