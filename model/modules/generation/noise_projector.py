import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


# L Block (there exists differences)
class ProjBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.one_conv = spectral_norm(
            nn.Conv2d(in_channels, out_channels - in_channels,
                      kernel_size=1, padding=0))
        self.double_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, padding=1)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(out_channels, out_channels,
                                    kernel_size=3, padding=1))
        )

    def forward(self, x):
        # x: [in_channels, H/32, W/32]

        # x1: [out_channels, H/32, W/32]
        x1 = torch.cat((x, self.one_conv(x)), dim=1)
        # x2: [out_channels, H/32, W/32]
        x2 = self.double_conv(x)

        return x1 + x2


class NoiseProjector(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels  # 32

        self.conv_first = spectral_norm(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1))
        self.L1 = ProjBlock(in_channels * 2, in_channels * 4)
        self.L2 = ProjBlock(in_channels * 4, in_channels * 8)
        self.L3 = ProjBlock(in_channels * 8, in_channels * 16)
        self.L4 = ProjBlock(in_channels * 16, in_channels * 32)

    def forward(self, x):
        # x: [32, H/32, W/32]

        x = self.conv_first(x)  # x: [64, H/32, W/32]
        x = self.L1(x)  # x: [128, H/32, W/32]
        x = self.L2(x)  # x: [256, H/32, W/32]
        x = self.L3(x)  # x: [512, H/32, W/32]
        x = self.L4(x)  # x: [1024, H/32, W/32]

        return x
