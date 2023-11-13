import datetime
import torch
from model import UNET
from train import gaussianNoise, NoisyEMNIST

model = UNET()
model.load_state_dict(torch.load('./denoiser.pth'))
model.eval()

testDataEMNIST = torchvision.datasets.EMNIST(root='./data', split='balanced', train=False, download=False, transform=transform)
noisyTestDataEMNIST = NoisyEMNIST(testDataEMNIST, gaussianNoise)

if not os.path.exists('testImages/'): os.makedirs('testImages/')
toPIL = transforms.ToPILImage()
for i in range(10):
    print(f"Iteration {i}: {datetime.datetime.now()}")
    # img = toPIL(testDataEMNIST.data[i])
    # img.save(f'testImages/image_{i}.png')
    # img = toPIL(noisyTestDataEMNIST[i][0])
    # img.save(f'testImages/image_{i}_noisy.png')
    print(model(noisyTestDataEMNIST[i][0]))
