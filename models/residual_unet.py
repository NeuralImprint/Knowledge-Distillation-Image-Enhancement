import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x))) + x

class ResidualUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(ResidualUNet, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(features)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(features, features*2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(features*2)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*2, features*4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(features*4)
        )
        self.up1 = nn.ConvTranspose2d(features*4, features*2, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.ReLU(inplace=True),
            ResidualBlock(features*2)
        )
        self.up2 = nn.ConvTranspose2d(features*2, features, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.ReLU(inplace=True),
            ResidualBlock(features)
        )
        self.final = nn.Conv2d(features, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d1 = self.dec1(self.up1(b) + e2)
        d2 = self.dec2(self.up2(d1) + e1)
        return self.final(d2)
