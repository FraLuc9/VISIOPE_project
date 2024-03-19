import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.conv3(residual)
        out = self.relu(out)
        return out


class OliveOil(nn.Module):
    def __init__(self):
        super().__init__()

        self.inconv = nn.Conv2d(3, 32, 3, padding=1)

        self.downconv1 = ResidualBlock(32, 64)

        self.downconv2 = ResidualBlock(64, 128)

        self.downconv3 = ResidualBlock(128, 256)

        self.downconv4 = ResidualBlock(256, 512)

        self.conv51 = nn.Conv2d(522, 1024, 3, 1, 1)
        self.conv52 = nn.Conv2d(1034, 512, 3, 1, 1)

        self.upconv1 = ResidualBlock(1024, 256)

        self.upconv2 = ResidualBlock(512, 128)

        self.upconv3 = ResidualBlock(256, 64)
        self.upconv4 = ResidualBlock(128, 32)

        self.outconv = nn.Conv2d(64, 3, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.upscale = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.bnn51 = nn.BatchNorm2d(1024)
        self.bnn52 = nn.BatchNorm2d(512)

        self.relu = nn.LeakyReLU()

        # change to 8 if 256x256
        self.cond = nn.Embedding(1024, 4)

    def forward(self, x, labels):
        # change to 8 if 256x256
        labconv = self.cond(labels.long()).unsqueeze(3).repeat(1, 1, 1, 4)

        xin = self.inconv(x)

        # downconv1
        x1 = self.pool(xin)
        x1 = self.downconv1(x1)

        # downconv2
        x2 = self.pool(x1)
        x2 = self.downconv2(x2)

        # downconv3
        x3 = self.pool(x2)
        x3 = self.downconv3(x3)

        # downconv4
        x4 = self.pool(x3)
        x4 = self.downconv4(x4)

        # papillon
        x5 = self.pool(x4)
        x5 = self.relu(self.bnn51(self.conv51(torch.cat((x5, labconv), dim=1))))
        x5 = self.relu(self.bnn52(self.conv52(torch.cat((x5, labconv), dim=1))))

        # x4 = self.conv42(torch.cat((x4, labconv), dim=1))

        # upconv1
        x6 = self.upscale(x5)
        x6 = torch.cat((x6, x4), dim=1)
        x6 = self.upconv1(x6)

        # upconv2
        x7 = self.upscale(x6)
        x7 = torch.cat((x7, x3), dim=1)
        x7 = self.upconv2(x7)

        # upconv3
        x8 = self.upscale(x7)
        x8 = torch.cat((x8, x2), dim=1)
        x8 = self.upconv3(x8)

        # upconv4
        x9 = self.upscale(x8)
        x9 = torch.cat((x9, x1), dim=1)
        x9 = self.upconv4(x9)

        x = self.upscale(x9)
        x = torch.cat((x, xin), dim=1)
        x = self.outconv(x)

        return x