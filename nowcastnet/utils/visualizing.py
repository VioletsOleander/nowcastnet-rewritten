import os

import torch
import matplotlib.pyplot as plt


def plot_and_save(frames: torch.Tensor, save_dir, vmin, vmax, cmap='viridis'):
    for frame_idx, frame in enumerate(frames):
        fig = plt.figure()
        ax = plt.axes()

        ax.set_axis_off()

        alpha = frame.copy()
        alpha[alpha < 1] = 0
        alpha[alpha > 1] = 1

        ax.imshow(frame, alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)

        plt.savefig(os.path.join(save_dir, f'frame-{frame_idx}.png'))
        plt.close()


def crop_frames(frames: torch.Tensor, crop_size):
    height = frames.shape[-2]
    width = frames.shape[-1]

    if crop_size == -1:
        return frames

    crop_length = crop_size // 2
    h_center = height // 2
    w_center = width // 2

    frames = frames[...,
                    h_center-crop_length:h_center+crop_length,
                    w_center-crop_length:w_center+crop_length]

    return frames
