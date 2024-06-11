import torch
from torch import nn, cat
from torch.nn.functional import pad


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.encoder = nn.ModuleList([
            EncodeDown(in_channels, 64),
            EncodeDown(64, 128),
            EncodeDown(128, 256),
            EncodeDown(256, 512),
            EncodeDown(512, 1024)
        ])

        self.double_conv = DoubleConv(1024, 2048)

        self.decoder = nn.ModuleList([
            DecodeUp(2048, 1024),
            DecodeUp(1024, 512),
            DecodeUp(512, 256),
            DecodeUp(256, 128),
            DecodeUp(128, 64)
        ])

        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for layer in self.encoder:
            x, pooled = layer(x)
            skips.append(x)
            x = pooled

        x = self.double_conv(x)

        for skip, layer in zip(reversed(skips), self.decoder):
            x = layer(x, skip)

        return self.output(x)


class EncodeDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.conv(x)
        pool = self.pool(x)
        return x, pool


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DecodeUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = cat([x, skip], dim=1)
        return self.conv(x)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class UNetOld(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetOld, self).__init__()

        self.down1 = EncodeDown(in_channels, 64)
        self.down2 = EncodeDown(64, 128)
        self.down3 = EncodeDown(128, 256)
        self.down4 = EncodeDown(256, 512)

        self.double_conv = DoubleConv(512, 1024)

        self.up1 = DecodeUp(1024, 512)
        self.up2 = DecodeUp(512, 256)
        self.up3 = DecodeUp(256, 128)
        self.up4 = DecodeUp(128, 64)

        self.output = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        s1, p1 = self.down1(x)
        s2, p2 = self.down2(p1)
        s3, p3 = self.down3(p2)
        s4, p4 = self.down4(p3)
        b = self.double_conv(p4)
        d1 = self.up1(b, s4)
        d2 = self.up2(d1, s3)
        d3 = self.up3(d2, s2)
        d4 = self.up4(d3, s1)
        output = self.output(d4)
        return output


class EncDec(nn.Module):
    def __init__(self, in_channels, out_channels, num_steps=10):
        super(EncDec, self).__init__()

        self.num_steps = num_steps

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.diffusion_steps = nn.ModuleList([
            ResNetBlock(256, 256) for _ in range(self.num_steps)
        ])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)

        for i in range(self.num_steps):
            encoded = self.diffusion_steps[i](encoded)

        decoded = self.decoder(encoded)
        return decoded
