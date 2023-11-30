import os, torch, torchvision
import torch.nn as nn
from model import UNET, Discriminator
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

num_epochs, batch_size = 10, 32
noisyTrainedData = DataLoader(noisyTrainDataEMNIST, batch_size=batch_size, shuffle=True)

generator = UNET()
discriminator = Discriminator()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001)
adversarial_loss = torch.nn.BCELoss()

l1_loss = nn.L1Loss()


for epoch in range(num_epochs):
    for i, (noisy_imgs, original_imgs) in enumerate(noisyTrainedData):
        valid = torch.ones((noisy_imgs.size(0), 1), requires_grad=False)
        fake = torch.zeros((noisy_imgs.size(0), 1), requires_grad=False)

        optimizer_G.zero_grad()

        gen_imgs = generator(noisy_imgs)
        g_adv_loss  = adversarial_loss(discriminator(gen_imgs), valid)

        g_l1_loss = l1_loss(gen_imgs, original_imgs)

        g_loss = g_adv_loss + g_l1_loss

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(original_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(noisyTrainedData)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

torch.save(generator.state_dict(), './generator.pth')
torch.save(discriminator.state_dict(), './discriminator.pth')