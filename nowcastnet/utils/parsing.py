import argparse
from dataclasses import dataclass
from pathlib import Path

import tomlkit
from tomlkit import TOMLDocument


@dataclass
class CommonConfig:
    # positional arguments
    dataset_path: str | None = None
    dataset_name: str = "radar"

    # data information arguments
    input_length: int = 9
    pred_length: int = 20
    image_height: int = 512
    image_width: int = 512

    # data loading and processing arguments
    cpu_workers: int = 0
    case_type: str = "normal"
    crop_size: int = 384
    batch_size: int = 1


@dataclass
class InferenceConfig(CommonConfig):
    # positional arguments
    weights_path: str | None = None
    results_path: str | None = None

    # model configuration arguments
    generator_base_channels: int = 32
    device: str = "cpu"

    # other configuration arguments
    config_path: str | None = None
    save_original_data: bool = True
    log_path: str = "inference.log"
    seed: int = 42

    # configurations derived from other arguments
    total_length: int | None = None
    generator_decoder_input_channels: int | None = None


@dataclass
class EvaluationConfig(CommonConfig):
    infer_results_path: str | None = None
    eval_results_path: str | None = None
    csi_thresholds: int = 16
    pooling_kernel_size: int = 2


def setup_parser(description) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    defaults = CommonConfig()

    # positional arguments (required if config_path is not provided)
    parser.add_argument(
        "dataset_path",
        type=str,
        nargs="?",
        default=defaults.dataset_path,
        help="path of the dataset",
    )

    # data information arguments (optional)
    data_info_group = parser.add_argument_group("data information arguments")
    data_info_group.add_argument(
        "--dataset_name",
        type=str,
        default=defaults.dataset_name,
        help="name of target dataset",
    )
    data_info_group.add_argument(
        "--input_length",
        type=int,
        default=defaults.input_length,
        help="number of input frames",
    )
    data_info_group.add_argument(
        "--pred_length",
        type=int,
        default=defaults.pred_length,
        help="number of frames to predict",
    )
    data_info_group.add_argument(
        "--image_height",
        type=int,
        default=defaults.image_height,
        help="height of input frames",
    )
    data_info_group.add_argument(
        "--image_width",
        type=int,
        default=defaults.image_width,
        help="width of input frames",
    )

    # data loading and processing arguments (optional)
    data_process_group = parser.add_argument_group(
        "data loading and processing arguments"
    )
    data_process_group.add_argument(
        "--cpu_workers",
        type=int,
        default=defaults.cpu_workers,
        help="num_workers for pytorch DataLoader",
    )
    data_process_group.add_argument(
        "--case_type",
        type=str,
        default=defaults.case_type,
        choices=["normal", "large"],
        help="different case_type corresponds to different image processing method for generated frames",
    )
    data_process_group.add_argument(
        "--crop_size",
        type=int,
        default=defaults.crop_size,
        help="size of the cropped frame predictions, -1 means do not crop the predictions",
    )
    data_process_group.add_argument(
        "--batch_size",
        type=int,
        default=defaults.batch_size,
        help="size of minibatch",
    )

    return parser


def parse_config_file(config_file_path: str) -> TOMLDocument:
    with Path(config_file_path).open("r", encoding="utf-8") as f:
        configs_from_file = tomlkit.load(f)

    return configs_from_file
