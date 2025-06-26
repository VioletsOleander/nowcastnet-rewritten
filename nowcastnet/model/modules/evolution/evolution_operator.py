import torch
import torch.nn as nn
import torch.nn.functional as F


class EvolutionOperator(nn.Module):
    def __init__(self, grid_shape, grid_device):
        super().__init__()

        self.coordinate_grid = self._create_coordinate_grid(
            grid_shape, grid_device)

    def _create_coordinate_grid(self, shape, device):
        batch_size, n_channels, height, width = shape

        horizontal_coordinates = torch.arange(0, width).view(
            1, -1).repeat(height, 1).to(device=device)
        vertical_coordinates = torch.arange(0, height).view(
            -1, 1).repeat(1, width).to(device=device)

        horizontal_coordinates = horizontal_coordinates.view(
            1, 1, height, width).repeat(batch_size, 1, 1, 1)
        vertical_coordinates = vertical_coordinates.view(
            1, 1, height, width).repeat(batch_size, 1, 1, 1)

        coordinate_grid = torch.cat(
            (horizontal_coordinates, vertical_coordinates), dim=1)

        return coordinate_grid

    def _normalize_coordinates(self, coordinates, size):
        return 2 * coordinates / max(size - 1, 1) - 1

    def _flow(self, x, motion_field: torch.Tensor, coordinate_grid: torch.Tensor,
              interpolation_mode, padding_mode):
        batch_size, n_channels, height, width = x.shape

        # reverse flow according to motion_field
        # coordinate_grid[:, :, h, w] = y, x indicates the partical in position (h, w)
        # in the next time frame is flowed from position (y, x) from the previous time frame
        coordinate_grid = coordinate_grid + motion_field

        # normalize horizontal coordinates values from [0, width-1] to [-1, 1]
        normalized_width_grid = self._normalize_coordinates(
            coordinate_grid[:, 0, :, :], width)
        # normalize vertical coordinates values from [0, height-1] to [-1, 1]
        normalized_height_grid = self._normalize_coordinates(
            coordinate_grid[:, 1, :, :], height)
        # concatenate normalized coordinates
        normalized_coordinate_grid = torch.stack(
            (normalized_width_grid, normalized_height_grid), dim=1)

        # [B, 2, H, W] -> [B, H, W, 2]
        normalized_coordinate_grid = normalized_coordinate_grid.permute(
            0, 2, 3, 1)

        x = F.grid_sample(input=x, grid=normalized_coordinate_grid,
                          mode=interpolation_mode, padding_mode=padding_mode, align_corners=True)

        return x

    def forward(self, x, motion_field, intensity_field,
                interpolation_mode='bilinear', padding_mode='zeros'):
        x = self._flow(x, motion_field, self.coordinate_grid,
                       interpolation_mode, padding_mode)
        x = x + intensity_field

        return x
