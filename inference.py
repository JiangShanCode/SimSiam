import torch
from torchvision import transforms
from PIL import Image

path_model = "./simsiam.pkl"
model = torch.load(path_model).encoder.to('cuda:0')
model.eval()
# print(model)

base_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

with open('/media/Data/LiveSR/Game_lol/HR_sub/1_s001.png', 'rb') as f:
    img = Image.open(f).convert('RGB')

img = base_transforms(img)
img = torch.unsqueeze(img,0).to('cuda:0')
# print(img.size())

print(model(img).size())