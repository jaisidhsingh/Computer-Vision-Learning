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
import random

# loading and preprocessing:
img = Image.open("snow.jpg")
ans = Image.open("snow-segm.png") 

transform_img = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor()])

img = transform_img(img)
img = img.view((1, 3, 32, 32))
ans = transform_img(ans)
ans = ans.view((1, 3, 32, 32))

print(img.shape)
print(ans.shape)


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
		x = self.ups1(x)
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		return x

net = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(10):
	runningLoss = 0.0
	optimizer.zero_grad()
	output = net(img)
	loss = criterion(output, ans)
	loss.backward()
	optimizer.step()
	runningLoss += loss.item()
	print("Epoch: ", epoch, " Loss: ", runningLoss)


out = net(img)
out = out.detach().numpy()
out = np.array(out).reshape((32, 32, 3))

plt.figure()
plt.imshow(out)
plt.show()
