import datetime, os, torch, torchvision
from model import UNET
from PIL import Image
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

model = UNET()
model.load_state_dict(torch.load('./denoiser.pth'))
model.eval()

transform = transforms.Compose([transforms.ToTensor(),])
testDataEMNIST = torchvision.datasets.EMNIST(root='./data', split='balanced', train=False, download=False, transform=transform)
noisyTestDataEMNIST = NoisyEMNIST(testDataEMNIST, gaussianNoise)
toPIL = transforms.ToPILImage()

if not os.path.exists('testImages/'): os.makedirs('testImages/')
for i in range(10):
    img = toPIL(testDataEMNIST.data[i])
    img.save(f'testImages/image_{i}.png')
    noisyImg = toPIL(noisyTestDataEMNIST[i][0])
    noisyImg.save(f'testImages/image_{i}_noisy.png')
    input_tensor = noisyTestDataEMNIST[i][0].unsqueeze(0)
    output = model(input_tensor)
    reconstructedImg = toPIL(output.squeeze())

    combinedImages = [img, noisyImg, reconstructedImg]
    combinedPath = f'testImages/image_{i}_gan.png'

    total_width = sum(image.width for image in combinedImages)
    max_height = max(image.height for image in combinedImages)
    collage = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for image in combinedImages:
        collage.paste(image, (x_offset, 0))
        x_offset += image.width
    collage.save(combinedPath)
