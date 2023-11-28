import datetime, os, torch, torchvision
from model import UNET
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
    print(f"Image {i}: {datetime.datetime.now()}")
    img = toPIL(testDataEMNIST.data[i])
    img.save(f'testImages/image_{i}.png')
    img = toPIL(noisyTestDataEMNIST[i][0])
    img.save(f'testImages/image_{i}_noisy.png')
    image_tensor = noisyTestDataEMNIST[i][0].unsqueeze(0)
    output = model(image_tensor)
    img = toPIL(output.squeeze())
    img.save(f'testImages/image_{i}_reconstructed.png')
