from argparse import Namespace

from nowcastnet.model.modules.evolution.evolution_network import EvolutionNetwork
from nowcastnet.model.modules.evolution.evolution_operator import EvolutionOperator
from nowcastnet.model.modules.generation.generative_encoder import GenerativeEncoder
from nowcastnet.model.modules.generation.generative_decoder import GenerativeDecoder
from nowcastnet.model.modules.generation.noise_projector import NoiseProjector

import torch
import torch.nn as nn


class NowcastNet(nn.Module):
    def __init__(self, configs: Namespace):
        super().__init__()

        self.configs = configs
        self.device = configs.device
        self.input_length = configs.input_length  # 9
        self.pred_length = configs.pred_length  # 20
        self.generator_base_channels = configs.generator_base_channels  # 32

        self.evo_net = EvolutionNetwork(
            in_channels=configs.input_length,
            n_classes=configs.pred_length,
            base_channels=32,
            bilinear=True)

        self.gen_enc = GenerativeEncoder(
            in_channels=configs.total_length,
            base_channels=configs.generator_base_channels)

        self.gen_dec = GenerativeDecoder(
            in_channels=configs.gen_decoder_input_channels,
            base_channels=configs.generator_base_channels,
            evo_channels=configs.pred_length,
            out_channels=configs.pred_length)

        self.proj = NoiseProjector(
            in_channels=configs.generator_base_channels)

        self.evo_operator = EvolutionOperator(grid_shape=(configs.batch_size, 1,
                                                          configs.image_height, configs.image_width),
                                              grid_device=configs.device)

    def forward(self, frames, noise):
        # frames: [batch_size, 9, H, W]
        batch_size, n_channels, height, width = frames.shape

        # Evolution Network
        # intensity_field: [batch_size, 20, H, W]
        # motion_field: [batch_size, 40, H, W]
        intensity_field, motion_field = self.evo_net(frames)

        # intensity_field: [batch_size, 20, 1, H, W]
        intensity_field = intensity_field.reshape(
            batch_size, self.pred_length, 1, height, width)
        # motion_field: [batch_size, 20, 2, H, W]
        motion_field = motion_field.reshape(
            batch_size, self.pred_length, 2, height, width)

        # Evolution Operator
        evolved_frames = []
        # latest_frame: [batch_size, 1, H, W]
        latest_frame = frames[:, self.input_length-1:self.input_length, :, :]
        for i in range(self.pred_length):
            latest_frame = self.evo_operator(
                x=latest_frame,
                # [batch_size, 1, H, W]
                motion_field=motion_field[:, i, :, :, :],
                # [batch_size, 1, H, W]
                intensity_field=intensity_field[:, i, :, :, :],
                interpolation_mode='nearest',
                padding_mode='border')
            evolved_frames.append(latest_frame)
        # evolved_frames: [batch_size, 20, H, W]
        evolved_frames = torch.cat(evolved_frames, dim=1) / 128

        # Generation Network
        # Nowcast Encoder
        # frames: [batch_size, 29, H, W]
        frames = torch.cat((frames, evolved_frames), dim=1)
        # evolution_feature: [batch_size, 256, H/8, W/8]
        evolution_feature = self.gen_enc(frames)

        # Noise Projector
        # noise: [batch_size, 32, H/32, W/32]
        # noise_feature: [batch_size, 1024, H/32, W/32]
        noise_feature = self.proj(noise)
        # noise_feature: [batch_size, ..., 4, 4, 8, 8]
        noise_feature = noise_feature.reshape(batch_size, -1, 4, 4, 8, 8)
        # noise_feature: [batch_size, ..., 8, 8, 4, 4]
        noise_feature = noise_feature.permute(0, 1, 4, 5, 2, 3)
        # noise_feature: [batch_size, 1024/16=64, H/8, W/8]
        noise_feature = noise_feature.reshape(
            batch_size, -1, height//8, width//8)

        # Nowcast Decoder
        # concated_feature: [batch_size, 256+64=320, H/8, W/8]
        concated_feature = torch.cat((evolution_feature, noise_feature), dim=1)
        # generated_frames: [batch_size, 20, H, W]
        generated_frames = self.gen_dec(concated_feature, evolved_frames)

        return generated_frames
