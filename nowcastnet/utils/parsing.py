import argparse


def setup_parser(description) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # positoinal args
    parser.add_argument("dataset_path", type=str, help="path of the dataset")

    # optional args
    # data information arguments
    data_info_group = parser.add_argument_group("data information arguments")
    data_info_group.add_argument(
        "--dataset_name", type=str, default="radar", help="name of target dataset"
    )
    data_info_group.add_argument(
        "--input_length",
        type=int,
        default=9,
        help="number of input frames for the model",
    )
    data_info_group.add_argument(
        "--total_length", type=int, default=29, help="number of total input frames"
    )
    data_info_group.add_argument(
        "--image_height", type=int, default=512, help="height of input frames"
    )
    data_info_group.add_argument(
        "--image_width", type=int, default=512, help="width of input frames"
    )

    # data loading and processing arguments
    data_process_group = parser.add_argument_group(
        "data loading and processing arguments"
    )
    data_process_group.add_argument(
        "--cpu_workers",
        type=int,
        default=0,
        help="number of working processes for data loading",
    )
    data_process_group.add_argument(
        "--case_type",
        type=str,
        default="normal",
        choices=["normal", "large"],
        help="different case_type corresponds to different image processing method for generated frames",
    )
    data_process_group.add_argument(
        "--crop_size",
        type=int,
        default=384,
        help="size of the cropped frame predictions, -1 means no cropping",
    )
    data_process_group.add_argument(
        "--batch_size", type=int, default=1, help="size of minibatch"
    )

    return parser
