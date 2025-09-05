import os

import matplotlib.pyplot as plt
import numpy as np


def plot_frames(
    frames: np.ndarray, save_dir: str, vmin: float, vmax: float, cmap="viridis"
):
    for frame_idx, frame in enumerate(frames):
        _ = plt.figure()
        ax = plt.axes()

        ax.set_axis_off()

        alpha = frame.clone()
        alpha[alpha < 1] = 0
        alpha[alpha > 1] = 1

        ax.imshow(frame, alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)

        plt.savefig(os.path.join(save_dir, f"frame-{frame_idx}.png"), dpi=300)
        plt.close()


def crop_frames(frames: np.ndarray, crop_size: int) -> np.ndarray:
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
    _ = plt.figure()
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
