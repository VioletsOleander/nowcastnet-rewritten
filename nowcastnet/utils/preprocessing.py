import argparse
import logging
import os

import numpy as np
from torch.utils.data import DataLoader

from nowcastnet.utils.visualizing import crop_frames, plot_frames


def preprocess(dataloader: DataLoader, configs: argparse.Namespace):
    logging.info('Preprocessing started')

    for batch, (observed_frames, future_frames) in enumerate(dataloader):
        logging.info(f'Batch: {batch}/{len(dataloader)}')

        observed_frames = observed_frames.detach().cpu().numpy()
        future_frames = future_frames.detach().cpu().numpy()

        if configs.case_type == 'normal':
            observed_frames = crop_frames(frames=observed_frames,
                                          crop_size=configs.crop_size)
            future_frames = crop_frames(frames=future_frames,
                                        crop_size=configs.crop_size)

        results_path = os.path.join(configs.path_to_preprocessed, str(batch))
        observed_save_dir = os.path.join(results_path, 'observed')
        future_save_dir = os.path.join(results_path, 'future')
        os.makedirs(observed_save_dir, exist_ok=True)
        os.makedirs(future_save_dir, exist_ok=True)

        plot_frames(frames=observed_frames[0],
                    save_dir=observed_save_dir,
                    vmin=0,
                    vmax=40)
        plot_frames(frames=future_frames[0],
                    save_dir=future_save_dir,
                    vmin=0,
                    vmax=40)

        if configs.save_original_data:
            np.save(os.path.join(
                observed_save_dir, 'frames.npy'), observed_frames[0])
            np.save(os.path.join(
                future_save_dir, 'frames.npy'), future_frames[0])

    logging.info('Preprocessing finished')
    logging.info(f'Results saved to {configs.path_to_preprocessed}')
