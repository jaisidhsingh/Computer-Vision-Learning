import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as TF
import matplotlib.pyplot as plt


arch_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CnnBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))


class YoloV1(nn.Module):
    def __init__(self, in_channels=3,  **kwargs):  # edit later
        super(YoloV1, self).__init__()
        self.arch = arch_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.arch)
        self.fc = self._create_fc(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fc(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, arch):
        layers = []
        in_channels = self.in_channels

        for x in arch:
            if type(x) == tuple:
                layers += [CnnBlock(in_channels, out_channels=x[1],
                                    kernel_size=x[0], stride=x[2], padding=x[3])]
                in_channels = x[1]

            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]

            elif type(x) == list:
                conv1 = x[0]  # tuple
                conv2 = x[1]  # tuple
                repeat_count = x[2]  # int

                for count in range(repeat_count):
                    layers += [CnnBlock(in_channels, conv1[1],
                                        kernel_size=conv1[0], stride=conv1[2], padding=conv1[3])]
                    layers += [CnnBlock(conv1[1], conv2[1], kernel_size=conv2[0],
                                        stride=conv2[2], padding=conv2[3])]
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fc(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),  # 4096 in the paper
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B*5)))  # to be reshaped into SxSx30, where C+B*5=30


def test(S=7, B=2, C=20):
    model = YoloV1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2, 3, 448, 448))
    y = model(x)
    print(y.shape)

# test()
