import torch.nn as nn


class DenoisingDiffusionNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, num_steps):
        super(DenoisingDiffusionNetwork, self).__init__()

        self.num_steps = num_steps

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        for _ in range(self.num_steps):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)

        return x