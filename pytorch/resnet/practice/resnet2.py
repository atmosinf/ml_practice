# resnet practice 2. might have mistakes

import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels=3, inter_channels=64, identity_downsample=None, stride=1):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv3 = nn.Conv2d(inter_channels, inter_channels * self.expansion, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inter_channels*self.expansion)
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
    def __init__(self, block, layers, img_channels, num_classes):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64, 1)
        self.layer2 = self._make_layer(block, layers[1], 128, 2)
        self.layer3 = self._make_layer(block, layers[2], 256, 2)
        self.layer4 = self._make_layer(block, layers[3], 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        print('shape after 1st conv', x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print('shape after 1st maxpool', x.shape)

        x = self.layer1(x)
        print('shape after 1st block', x.shape)
        x = self.layer2(x)
        print('shape after 2nd block', x.shape)
        x = self.layer3(x)
        print('shape after 3rd block', x.shape)
        x = self.layer4(x)
        print('shape after 4th block', x.shape)

        x = self.avgpool(x)
        print('shape after avgpool', x.shape)
        x = x.reshape(x.shape[0], -1)
        print('shape after reshape before fc', x.shape)
        x = self.fc(x)
        print('shape of the output', x.shape)

        return x

    def _make_layer(self, block, num_residual_layers, inter_channels, stride):
        layers = []

        if stride !=1 or self.in_channels != inter_channels*4:
            identity_downsample = nn.Sequential(
                                    nn.Conv2d(self.in_channels, inter_channels*4, kernel_size=1, padding=0, stride=stride, bias=False),
                                    nn.BatchNorm2d(inter_channels*4)
                                    )
            layers.append(block(self.in_channels, inter_channels, identity_downsample, stride))

        self.in_channels = inter_channels * 4
        print('self.in_channels after mult:', self.in_channels)

        for _ in range(num_residual_layers - 1):
            layers.append(block(self.in_channels, inter_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(Block, layers=[3,4,6,3], img_channels=img_channels, num_classes=num_classes)

def test():
    model = ResNet50(3, 1000)
    sampinput = torch.randn((4,3,1000,1000))
    out = model(sampinput)
    print(out.shape)

test()

# model = ResNet50(3, 1000)
# print(model)