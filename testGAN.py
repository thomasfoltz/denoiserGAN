import datetime, os, torch, torchvision
from model import UNET, Discriminator
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

generator = UNET()
generator.load_state_dict(torch.load('./generator.pth'))
generator.eval()

discriminator = Discriminator()
discriminator.load_state_dict(torch.load('./discriminator.pth'))
discriminator.eval()

transform = transforms.Compose([transforms.ToTensor(),])
testDataEMNIST = torchvision.datasets.EMNIST(root='./data', split='balanced', train=False, download=False, transform=transform)
noisyTestDataEMNIST = NoisyEMNIST(testDataEMNIST, gaussianNoise)
toPIL = transforms.ToPILImage()

if not os.path.exists('results/'): os.makedirs('results/')
for i in range(20):
    img = toPIL(testDataEMNIST.data[i])
    img.save(f'results/image_{i+1}.png')
    noisyImg = toPIL(noisyTestDataEMNIST[i][0])
    noisyImg.save(f'results/image_{i+1}_noisy.png')
    input_tensor = noisyTestDataEMNIST[i][0].unsqueeze(0)
    output = generator(input_tensor)
    reconstructedImg = toPIL(output.squeeze())
    print(f'image_{i+1} GAN conf: {round(discriminator(output).item()*100, 2)}%')

    combinedImages = [img, noisyImg, reconstructedImg]
    combinedPath = f'results/image_{i+1}_gan.png'

    total_width = sum(image.width for image in combinedImages)
    max_height = max(image.height for image in combinedImages)
    collage = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for image in combinedImages:
        collage.paste(image, (x_offset, 0))
        x_offset += image.width
    collage.save(combinedPath)
