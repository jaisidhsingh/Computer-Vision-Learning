import torch
import numpy as np
import PIL.Image as Image
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2


img = Image.open("cat01.jpg") 

transform_img = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor()])

img = transform_img(img)
img = img.view((1, 3, 32, 32))
print(img.shape)

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 3, 5)
		self.conv2 = nn.Conv2d(3, 3, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.ups1 = nn.Upsample(scale_factor=2, mode="nearest")

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.ups1(x)
		x = self.ups1(x)
		x = x.view((1, 3, 20, 20))
		return x

net = Net()
out = net.forward(img)
out = out.detach().numpy()
out = np.array(out).reshape(20, 20, 3)

plt.figure()
plt.imshow(out)
plt.show()