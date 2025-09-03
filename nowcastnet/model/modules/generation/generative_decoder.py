import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


# Spatial Norm
class SPADE(nn.Module):
    def __init__(self, feature_map_channels, label_map_channels):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(
            num_features=feature_map_channels, affine=False
        )

        hidden_channels = 64
        kernel_size = 3
        padding_size = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding_size = padding_size

        self.mlp_shared = nn.Sequential(
            nn.ReflectionPad2d(padding=padding_size),
            nn.Conv2d(label_map_channels, hidden_channels, kernel_size, padding=0),
            nn.ReLU(),
        )
        self.pad = nn.ReflectionPad2d(padding=padding_size)
        self.mlp_gamma = nn.Conv2d(
            hidden_channels, feature_map_channels, kernel_size, padding=0
        )
        self.mlp_beta = nn.Conv2d(
            hidden_channels, feature_map_channels, kernel_size, padding=0
        )

    def forward(self, x, condition):
        # instance Norm
        x = self.param_free_norm(x)

        # extract spatially adaptive affine transformation parameters
        condition = F.adaptive_avg_pool2d(condition, output_size=x.shape[2:])
        condition = self.mlp_shared(condition)
        gamma = self.mlp_gamma(self.pad(condition))
        beta = self.mlp_beta(self.pad(condition))

        # spatially adapative affine transformation (with residual shortcut)
        x = x * (1 + gamma) + beta

        return x


# S Block
class GenBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, evo_channels, dilation=1, double_conv=False
    ):
        super().__init__()

        mid_channels = min(in_channels, out_channels)
        self.evo_channels = evo_channels
        self.shortcut = in_channels != out_channels
        self.dilation = dilation
        self.double_conv = double_conv

        self.pad = nn.ReflectionPad2d(padding=dilation)
        self.conv_0 = spectral_norm(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, padding=0, dilation=dilation
            )
        )
        self.conv_1 = spectral_norm(
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=3, padding=0, dilation=dilation
            )
        )
        if self.shortcut:
            self.conv_s = spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.norm_0 = SPADE(
            feature_map_channels=in_channels, label_map_channels=evo_channels
        )
        self.norm_1 = SPADE(
            feature_map_channels=mid_channels, label_map_channels=evo_channels
        )
        if self.shortcut:
            self.norm_s = SPADE(
                feature_map_channels=in_channels, label_map_channels=evo_channels
            )

    def residual_shortcut(self, x, evo_output):
        if self.shortcut:
            x = self.conv_s(self.norm_s(x, evo_output))

        return x

    def activate(self, x):
        return F.leaky_relu(x, negative_slope=2e-1)

    def forward(self, x, evo):
        x_s = self.residual_shortcut(x, evo)
        x = self.conv_0(self.pad(self.activate(self.norm_0(x, evo))))
        if self.double_conv:
            x = self.conv_1(self.pad(self.activate(self.norm_1(x, evo))))

        x = x + x_s

        return x


class GenerativeDecoder(nn.Module):
    def __init__(self, in_channels, base_channels, evo_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels  # 320
        self.base_channels = base_channels  # 32
        self.evo_channels = evo_channels  # 20
        self.out_channels = out_channels  # 20

        self.fc = nn.Conv2d(in_channels, base_channels * 8, kernel_size=3, padding=1)

        self.head_0 = GenBlock(
            base_channels * 8, base_channels * 8, evo_channels, double_conv=False
        )
        self.G_middle_0 = GenBlock(
            base_channels * 8, base_channels * 4, evo_channels, double_conv=True
        )
        self.G_middle_1 = GenBlock(
            base_channels * 4, base_channels * 4, evo_channels, double_conv=True
        )

        self.up_0 = GenBlock(
            base_channels * 4, base_channels * 2, evo_channels, double_conv=False
        )
        self.up_1 = GenBlock(
            base_channels * 2, base_channels * 1, evo_channels, double_conv=True
        )
        self.up_2 = GenBlock(
            base_channels * 1, base_channels * 1, evo_channels, double_conv=True
        )

        self.conv_img = nn.Conv2d(
            base_channels * 1, out_channels, kernel_size=3, padding=1
        )

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x, evo):
        # x: [320, H/8, W/8], evo: [20, H, W]
        x = self.fc(x)  # x: [256, H/8, W/8]
        x = self.head_0(x, evo)  # x: [256, H/8, W/8]

        x = self.up(x)  # x: [256, H/4, W/4]
        x = self.G_middle_0(x, evo)  # x: [128, H/4, W/4]
        x = self.G_middle_1(x, evo)  # x: [128, H/4, W/4]

        x = self.up(x)  # x: [128, H/2, W/2]
        x = self.up_0(x, evo)  # x: [64, H/2, W/2]

        x = self.up(x)  # x: [64, H, W]
        x = self.up_1(x, evo)  # x: [32, H, W]
        x = self.up_2(x, evo)  # x: [32, H, W]

        x = self.conv_img(F.leaky_relu(x, negative_slope=2e-1))  # x: [20, H, W]

        return x
