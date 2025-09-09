import logging
from pathlib import Path
from typing import cast

import numpy as np
from torch.utils.data import DataLoader

from nowcastnet.utils.parsing import EvaluationConfig
from nowcastnet.utils.visualizing import crop_frames, plot_frames


def preprocess(dataloader: DataLoader, configs: EvaluationConfig):
    logging.info("Preprocessing started")

    for batch, (observed_frames, future_frames) in enumerate(dataloader):
        logging.info(f"Batch: {batch}/{len(dataloader)}")

        observed_frames = observed_frames.detach().cpu().numpy()
        future_frames = future_frames.detach().cpu().numpy()

        if configs.case_type == "normal":
            observed_frames = crop_frames(
                frames=observed_frames, crop_size=configs.crop_size
            )
            future_frames = crop_frames(
                frames=future_frames, crop_size=configs.crop_size
            )

        results_path = Path(cast(str, configs.preprocessed_dataset_path)) / str(batch)
        observed_save_dir = results_path / "observed"
        future_save_dir = results_path / "future"
        observed_save_dir.mkdir(parents=True, exist_ok=True)
        future_save_dir.mkdir(parents=True, exist_ok=True)

        plot_frames(
            frames=observed_frames[0], save_dir=observed_save_dir, vmin=0, vmax=40
        )
        plot_frames(frames=future_frames[0], save_dir=future_save_dir, vmin=0, vmax=40)

        if configs.save_original_data:
            np.save(observed_save_dir / "frames.npy", observed_frames[0])
            np.save(future_save_dir / "frames.npy", future_frames[0])

    logging.info("Preprocessing finished")
    logging.info(f"Results saved to {configs.preprocessed_dataset_path}")
