from PIL import Image
from torchvision import transforms

train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

with open("/media/Data/LiveSR/Game_lol_11/Val/LRbicx4_sub/1_s001.png", 'rb') as f:
    img = Image.open(f).convert('RGB')

    if train_transforms is not None:
        img = train_transforms(img)
        print(img.size())
