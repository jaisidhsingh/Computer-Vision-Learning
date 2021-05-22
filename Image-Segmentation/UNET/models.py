import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np

#Carvana kept in mind
class DoubleConv(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(DoubleConv, self).__init__()
        self.convBlock = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannels, outChannels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.convBlock(x)

class Unet( nn.Module):
    def __init__(self, inChannels=3, outChannels=1, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        #downsampling
        for feature in features:
            self.downs.append(DoubleConv(inChannels, feature))
            inChannels = feature
        
        #upsampling
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
        
        #bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        #final convolution layer
        self.finalConv = nn.Conv2d(features[0], outChannels, kernel_size=1)

    def forward(self, x):
        skipConns = []
        #forward the downsampling part
        for down in self.downs:
            x = down(x)
            skipConns.append(x)
            x = self.pooling(x)
        
        #bottleneck forward 
        x = self.bottleneck(x)
        skipConns = skipConns[::-1]

        #forward the upsampling part
        for i in range(0, len(self.ups), 2): # as we are upsampling, then passing a double conv, hence a step size of 2
            x = self.ups[i](x)
            skip = skipConns[i//2]

            #crop x if the shapes do not match
            if x.shape!=skip.shape:
                x = TF.resize(x, size=skip.shape[2:])

            concatSkip = torch.cat((skip, x), dim=1)
            x = self.ups[i+1](concatSkip)
        
        return self.finalConv(x)

def test():
    x = torch.randn((3, 1, 160, 160))
    model = Unet(inChannels=1, outChannels=1)
    prediction = model(x)
    print(x.shape, prediction.shape)

if __name__ == '__main__':
    test()
