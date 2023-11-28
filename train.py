import datetime, os, torch, torchvision
import torch.nn as nn
from model import UNET
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

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
trainDataEMNIST = torchvision.datasets.EMNIST(root='./data', split='balanced', train=True, download=False, transform=transform)
noisyTrainDataEMNIST = NoisyEMNIST(trainDataEMNIST, gaussianNoise)

model = UNET()
num_epochs, batch_size = 10, 32
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
noisyTrainedData = DataLoader(noisyTrainDataEMNIST, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    print(f"Iteration {epoch}: {datetime.datetime.now()}")
    model.train()
    running_loss = 0.0
    for noisy_imgs, original_imgs in noisyTrainedData:
        optimizer.zero_grad()
        outputs = model(noisy_imgs)
        loss = criterion(outputs, original_imgs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(noisyTrainedData)}")

torch.save(model.state_dict(), './denoiser.pth')