import argparse
import logging
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nowcastnet.datasets.factory import dataset_provider
from nowcastnet.model.nowcastnet import NowcastNet
from nowcastnet.utils.logging import log_configs, setup_logging
from nowcastnet.utils.parsing import InferenceConfig, parse_config_file, setup_parser
from nowcastnet.utils.visualizing import crop_frames, plot_frames


def refine_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    defaults = InferenceConfig()

    # positional arguments (required if config_path is not provided)
    parser.add_argument(
        "weights_path",
        type=str,
        nargs="?",
        default=defaults.weights_path,
        help="path of the pre-trained model weights",
    )
    parser.add_argument(
        "results_path",
        type=str,
        nargs="?",
        default=defaults.results_path,
        help="path to store the generated results",
    )

    # model configuration arguments (optional)
    model_group = parser.add_argument_group("model configuration arguments")
    model_group.add_argument(
        "--generator_base_channels",
        type=int,
        default=defaults.generator_base_channels,
        help="number of generator base channels",
    )
    model_group.add_argument(
        "--device", type=str, default=defaults.device, help="device to run the model"
    )

    # other configuration arguments (optional)
    other_group = parser.add_argument_group("other configuration arguments")
    other_group.add_argument(
        "--config_path",
        type=str,
        default=defaults.config_path,
        help="path of the config file, if provided, other command line arguments will be ignored",
    )
    other_group.add_argument(
        "--save_original_data",
        type=bool,
        default=defaults.save_original_data,
        help="whether to save the inferenced original numpy ndarray data of the inference result",
    )
    other_group.add_argument(
        "--log_path",
        type=str,
        default=defaults.log_path,
        help="path to store the log file",
    )
    other_group.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="random seed for reproducibility",
    )

    return parser


def prepare_configs(parser: argparse.ArgumentParser) -> InferenceConfig:
    # if config_path is provided, parse the config file
    # else use the command line arguments

    args = parser.parse_args()
    configs = InferenceConfig()

    if args.config_path is not None:
        configs_from_file = parse_config_file(args.config_path)
        for key, value in configs_from_file.items():
            if not hasattr(configs, key):
                raise ValueError(f"Unknown config key: {key}")
            setattr(configs, key, value)
    else:
        for key, value in vars(args).items():
            setattr(configs, key, value)

    assert configs.dataset_path is not None, "dataset_path should be provided."
    assert configs.weights_path is not None, "weights_path should be provided."
    assert configs.results_path is not None, "results_path should be provided."

    configs.total_length = configs.input_length + configs.pred_length
    configs.generator_decoder_input_channels = configs.generator_base_channels * 10

    return configs


def inference(model: nn.Module, dataloader: DataLoader, configs: InferenceConfig):
    logging.info("Inference started")

    np.random.seed(configs.seed)

    results_dir = Path(cast(str, configs.results_path))
    results_dir.mkdir(parents=True, exist_ok=True)

    model.to(device=configs.device)
    model.eval()

    for batch, (observed_frames, _) in enumerate(dataloader):
        logging.info(f"Batch: {batch + 1}/{len(dataloader)}")

        observed_frames = observed_frames.to(device=configs.device)
        noise = np.random.randn(
            configs.batch_size,
            configs.generator_base_channels,
            configs.image_height // 32,
            configs.image_width // 32,
        ).astype(np.float32)
        noise = torch.from_numpy(noise).to(device=configs.device)

        with torch.no_grad():
            predicted_frames = model(observed_frames, noise)
        predicted_frames = predicted_frames.detach().cpu().numpy()

        result_path = results_dir / str(batch)
        result_path.mkdir(parents=True, exist_ok=True)

        if configs.case_type == "normal":
            predicted_frames = crop_frames(
                frames=predicted_frames, crop_size=configs.crop_size
            )

        plot_frames(frames=predicted_frames[0], save_dir=result_path, vmin=1, vmax=40)

        if configs.save_original_data:
            np.save(result_path / "frames.npy", predicted_frames[0])

    logging.info("Inference finished")
    logging.info(f"Results saved to {results_dir.absolute()}")
    logging.info(f"Log saved to {Path(configs.log_path).absolute()}")


if __name__ == "__main__":
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    parser = refine_parser(setup_parser(description="Run NowcastNet inference"))
    configs = prepare_configs(parser)

    setup_logging(configs.log_path)
    log_configs(configs)

    model = NowcastNet(configs)
    model.load_state_dict(
        torch.load(cast(str, configs.weights_path), map_location=configs.device)
    )
    logging.info(
        f"Model weights loaded from {Path(cast(str, configs.weights_path)).absolute()}"
    )

    dataloader = dataset_provider(configs)
    logging.info(
        f"DataLoader created from {Path(cast(str, configs.dataset_path)).absolute()}"
    )

    inference(model, dataloader, configs)
