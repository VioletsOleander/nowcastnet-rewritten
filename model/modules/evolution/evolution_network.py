from model.modules.shared_modules import DoubleConv, Up, Down

import torch
import torch.nn as nn


# 1x1 Conv
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)

        return x


class EvolutionNetwork(nn.Module):
    def __init__(self, in_channels, n_classes, base_channels, bilinear):
        super().__init__()

        self.in_channels = in_channels  # 9
        self.n_classes = n_classes  # 20
        self.base_channels = base_channels  # 32
        self.bilinear = bilinear  # True

        # bilinear upsampling layer can not adjust channels
        # transposed convolution layer can adjust channels
        # Therefore, convolution layer in bilinear Upsample Block should adjust channels additionally
        factor = 2 if bilinear else 1

        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        # Intensity Decoder
        self.up1 = Up(base_channels * 16,
                      base_channels * 8 // factor,
                      bilinear)
        self.up2 = Up(base_channels * 8,
                      base_channels * 4 // factor,
                      bilinear)
        self.up3 = Up(base_channels * 4,
                      base_channels * 2 // factor,
                      bilinear)
        self.up4 = Up(base_channels * 2,
                      base_channels * 1,
                      bilinear)
        self.outc = OutConv(base_channels * 1, n_classes)
        self.gamma = nn.Parameter(torch.zeros(
            1, n_classes, 1, 1), requires_grad=True)

        # Motion Decoder
        self.up1_v = Up(base_channels * 16,
                        base_channels * 8 // factor,
                        bilinear)
        self.up2_v = Up(base_channels * 8,
                        base_channels * 4 // factor,
                        bilinear)
        self.up3_v = Up(base_channels * 4,
                        base_channels * 2 // factor,
                        bilinear)
        self.up4_v = Up(base_channels * 2,
                        base_channels * 1,
                        bilinear)
        self.outc_v = OutConv(base_channels * 1, n_classes * 2)

    def forward(self, x):
        # x: [9, H, W]

        # Encoding
        x1 = self.inc(x)  # x1: [32, H, W]
        x2 = self.down1(x1)  # x2: [64, H/2, W/2]
        x3 = self.down2(x2)  # x3: [128, H/4, H/4]
        x4 = self.down3(x3)  # x4: [256, H/8, W/8]
        x5 = self.down4(x4)  # x5: [256, H/16, W/16]

        # Intensity Decoding
        x = self.up1(x5, x4)  # x: [128, H/8, W/8]
        x = self.up2(x, x3)  # x: [64, H/4, H/4]
        x = self.up3(x, x2)  # x: [32, H/2, W/2]
        x = self.up4(x, x1)  # x: [32, H, W]
        x = self.outc(x) * self.gamma  # x: [20, H, W]

        # Motion Decoding
        v = self.up1_v(x5, x4)  # v: [128, H/8, W/8]
        v = self.up2_v(v, x3)  # v: [64, H/4, W/4]
        v = self.up3_v(v, x2)  # v: [32, H/2, W/2]
        v = self.up4_v(v, x1)  # v: [20, H, W]
        v = self.outc_v(v)  # v: [40, H, W]

        return x, v
