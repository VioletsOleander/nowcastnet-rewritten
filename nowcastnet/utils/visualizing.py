import os

import torch
import matplotlib.pyplot as plt


def plot_frames(frames: torch.Tensor, save_dir, vmin, vmax, cmap="viridis"):
    for frame_idx, frame in enumerate(frames):
        fig = plt.figure()
        ax = plt.axes()

        ax.set_axis_off()

        alpha = frame.copy()
        alpha[alpha < 1] = 0
        alpha[alpha > 1] = 1

        ax.imshow(frame, alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)

        plt.savefig(os.path.join(save_dir, f"frame-{frame_idx}.png"), dpi=300)
        plt.close()


def crop_frames(frames: torch.Tensor, crop_size):
    height = frames.shape[-2]
    width = frames.shape[-1]

    if crop_size == -1:
        return frames

    crop_length = crop_size // 2
    h_center = height // 2
    w_center = width // 2

    frames = frames[
        ...,
        h_center - crop_length : h_center + crop_length,
        w_center - crop_length : w_center + crop_length,
    ]

    return frames


def plot_line(x, y, x_ticks, y_ticks, x_label, y_label, title, save_dir, image_name):
    fig = plt.figure()
    ax = plt.axes()

    ax.plot(x, y)

    ax.set_xlabel(x_label)
    ax.set_xlim(0, len(x))
    ax.set_xticks(x_ticks)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 1)
    ax.set_yticks(y_ticks)

    ax.set_title(title)

    plt.savefig(os.path.join(save_dir, image_name), dpi=300)
    plt.close()
