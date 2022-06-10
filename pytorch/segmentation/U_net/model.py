'''
there is a difference between this implementation and the implementation in the paper. our model uses padded convolutions, while the paper uses valid convolutions.
the dataloading part will be much more difficult if we use valid convolutions.
'''
from audioop import reverse
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), # bias is unnecessary because it will be cancelled by the batchnorm anyway
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # up part of UNET
        for feature in reversed(features): # i think we can also use features[::-1] to get it reversed
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        # bottom part
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        #final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            self.pool(x)

        x = self.bottleneck(x) # bottom layer
        
        skip_connections = skip_connections[::-1] # reverse the order for the up part

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # does the ConvTranspose2d step in up
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape: # if the input is an odd numbered size, then after max pooling in the down layer, the value will be floored. i.e, 161x161 after max pool will become 80x80. this will cause a mismatch in size between the up and down layers and concat will not work
                x = TF.resize(x, size=skip_connection.shape[2:]) # pad x to match the shape of the height and width of skip_connection 

            concat_skip = torch.cat((skip_connection, x), dim=1) # add them at the channel dimension. remember that the dims are batch x channel x height x width.
            x = self.ups[idx+1](concat_skip) # does the DoubleConv step in up
        
        x = self.final_conv(x) # final layer

        return x

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape, x.shape) # input shape and output shape should match
    assert preds.shape == x.shape

if __name__ == '__main__':
    test()