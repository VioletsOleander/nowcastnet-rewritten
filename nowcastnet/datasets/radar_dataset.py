from pathlib import Path

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset


class RadarDataset(Dataset):
    def __init__(self, dataset_config: dict):
        super().__init__()
        self.input_data_type = dataset_config["input_data_type"]
        self.output_data_type = dataset_config["output_data_type"]

        self.image_width = dataset_config["image_width"]
        self.image_height = dataset_config["image_height"]
        # ensure the frames' height and width are limited
        assert self.image_height <= 1024 and self.image_width <= 1024

        self.pred_length = dataset_config["pred_length"]
        self.input_length = dataset_config["input_length"]
        self.total_length = self.input_length + self.pred_length

        self.dataset_path = Path(dataset_config["dataset_path"])

        self.sample_list = self._build_sample_list()

    def _build_sample_list(self):
        # each sample in sample_list is a list containing paths to its revelent frames
        sample_list = []
        for sample_dir in self.dataset_path.iterdir():
            if sample_dir.is_dir():
                # each case_dir contains 29 frames
                frame_paths = [
                    sample_dir / f"{sample_dir.name}-{str(i).zfill(2)}.png"
                    for i in range(29)
                ]
                sample_list.append(frame_paths)

        return sample_list

    def _load_frames(self, sample_idx) -> np.ndarray:
        frames = []
        frame_paths = self.sample_list[sample_idx]
        # load revelent frames for this sample
        for frame_path in frame_paths:
            frame = cv.imread(frame_path, cv.IMREAD_UNCHANGED)
            if frame is None:
                raise ValueError(f"Failed to load frame from {frame_path}")
            frames.append(np.expand_dims(frame, axis=0))
        sample = np.concatenate(frames, axis=0).astype(self.input_data_type) / 10 - 3

        assert (
            sample.shape[1] == self.image_height and sample.shape[2] == self.image_width
        )

        return sample

    def __getitem__(self, sample_idx) -> tuple[np.ndarray, np.ndarray]:
        sample = self._load_frames(sample_idx)
        # extract the latest self.total_length frames
        sample = sample[-self.total_length :]

        # mask = np.ones_like(sample)
        # mask[sample < 0] = 0

        sample = np.clip(sample, a_min=0, a_max=128)
        # sample = np.stack((sample, mask), axis=-1)

        return sample[: self.input_length], sample[self.input_length :]

    def __len__(self):
        return len(self.sample_list)
