import torch
from torchvision import datasets, transforms
import numpy as np

# Define the transformation
transform = transforms.Compose([transforms.ToTensor()])

# Load the dataset
train_dataset = datasets.EMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST(root='./data', train=False, download=True, transform=transform)

# Introduce noise
def add_noise(data, noise_factor=0.5):
    data = data + noise_factor * torch.randn(*data.shape)
    return torch.clamp(data, 0., 1.)

noisy_train_data = add_noise(train_dataset.data.float() / 255.)
noisy_test_data = add_noise(test_dataset.data.float() / 255.)
