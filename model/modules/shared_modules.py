import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


# (BN -> Conv -> Spectral Norm)
def conv_block(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.BatchNorm2d(num_features=in_channels),
        spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, padding=kernel_size//2))
    )


# (BN -> ReLU -> Conv -> Spectral Norm)
def conv_relu_block(in_channels, out_channels, kernel_size):
    return nn.Sequential(
        nn.BatchNorm2d(num_features=in_channels),
        nn.ReLU(inplace=True),
        spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, padding=kernel_size//2))
    )


# D Block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            conv_relu_block(in_channels, mid_channels, kernel_size),
            conv_relu_block(mid_channels, out_channels, kernel_size),
        )
        self.single_conv = conv_block(in_channels, out_channels, kernel_size)

    def forward(self, x):
        shortcut = self.single_conv(x)
        x = self.double_conv(x)

        return x + shortcut


# Down + D Block
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels, kernel_size)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)

        return x


# Upsample + D Block
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, kernel_size=3):
        super().__init__()

        if bilinear:
            # bilinear interpolation upsampling
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, kernel_size, in_channels//2)
        else:
            # transposed convolution upsampling
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # padding if necessary
        diff_y = x2.shape[2] - x1.shape[2]
        diff_x = x2.shape[3] - x1.shape[3]
        x1 = F.pad(x1, (diff_x//2, diff_x - diff_x//2,
                        diff_y//2, diff_y - diff_y//2))

        x = torch.cat((x2, x1), dim=1)
        x = self.conv(x)

        return x
