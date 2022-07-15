# resnet practice 1. might have mistakes

import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels=3, inter_channels=64, identity_downsample=None, stride=1):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, padding=0, stride=1, bias=False  )
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv3 = nn.Conv2d(inter_channels, inter_channels*self.expansion, kernel_size=1, padding=0, stride=1, bias=False)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], inter_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], inter_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], inter_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], inter_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, inter_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or inter_channels != self.in_channels * 4:
            identity_downsample = nn.Sequential(
                                    nn.Conv2d(self.in_channels, inter_channels*4, kernel_size=1, padding=0, stride=stride),
                                    nn.BatchNorm2d(inter_channels*4)
                                    )

        layers.append(block(self.in_channels, inter_channels, identity_downsample, stride))

        self.in_channels = inter_channels * 4

        for _ in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, inter_channels)) # stride = 1, even for the last 3 blocks, for every iteration of the block except the first iteration

        return nn.Sequential(*layers)
