"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as tf
from glob import glob
from natsort import natsorted
from basicsr.utils import bgr2ycbcr, scandir

class mydata(Dataset): 
    def __init__(self, root="/root/workspace/LiveSR/datasets/LiveSR/GamingDataSet/Train/Bicubic/x4_sub/", transform=None):
        super(mydata, self).__init__()
        self.root = root
        self.transform = transform 
        # self.resize = tf.Resize(256)
        # self.tf2tr = tf.ToTensor()
        self.imgs_path = natsorted(list(scandir(self.root, recursive=True, full_path=False)))
        self.imgs = []
        for path in self.imgs_path:
            with open(os.path.join(root,path), 'rb') as f:
                img = Image.open(f).convert('RGB')
                self.imgs.append(img)
        print("dataloder done")

    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, index):
        img = self.imgs[index]
    
        if self.transform is not None:
            img = self.transform(img)

        # print(out)
        return img

    # def get_image(self, index):
    #     path = self.imgs[index]
    #     with open(path, 'rb') as f:
    #         img = Image.open(f)
    #     # img = self.resize(img) 
    #     return img
