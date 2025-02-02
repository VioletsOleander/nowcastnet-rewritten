from model.modules.shared_modules import DoubleConv, Down

import torch.nn as nn


class GenerativeEncoder(nn.Module):
    def __init__(self, in_channels, base_channels):
        super().__init__()
        self.in_channels = in_channels  # 29
        self.base_channels = base_channels  # 32

        self.inc = DoubleConv(in_channels, base_channels, kernel_size=3)
        self.down1 = Down(base_channels * 1, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)

    def forward(self, x):
        # x: [29, H, W]
        x = self.inc(x)  # x: [32, H, W]
        x = self.down1(x)  # x: [64, H/2, W/2]
        x = self.down2(x)  # x: [128, H/4, W/4]
        x = self.down3(x)  # x: [256, H/8, H/8]

        return x
