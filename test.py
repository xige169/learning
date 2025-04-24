import torchvision
import torch
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_target = {
    0 : 'plane',
    1 : 'automobile',
    2 : 'bird',
    3 : 'cat',
    4 : 'deer',
    5 : 'dog',
    6 : 'frog',
    7 : 'horse',
    8 : 'ship',
    9 : 'truck'
}
img_pth = 'picture/ship.png'
img_PIL = Image.open(img_pth)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor()
])

img = transform(img_PIL)
img = img.reshape([1,-1,32,32])
# print(img.shape)
img = img.to(device)

module = torch.load('tudui_9.pth', weights_only=False).to(device)
# print(module)
module.eval()
with torch.no_grad():
    output = module(img)
    target = output.argmax(1).item()
# print(output)


print(f'picture is about: {data_target.get(target)}')