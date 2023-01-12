import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as tf

img = cv2.imread("/root/workspace/LiveSR/datasets/LiveSR/GamingDataSet/Train/HR/CSGO/CSGO_2.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(type(Image.fromarray(np.uint8(img_rgb))))
img_pil = Image.open("/root/workspace/LiveSR/datasets/LiveSR/GamingDataSet/Train/HR/CSGO/CSGO_2.png").convert('RGB')
tf2tr = tf.ToTensor()
# print(type(img),img.shape)
# print(img)
print(type(img_pil))
# print(np.asarray(img_pil))
# print(tf2tr(img))
# print(tf2tr(img_rgb))
print(tf2tr(img_pil))
