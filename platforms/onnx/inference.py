"""Do ONNX model Inference."""

import os
import logging

import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader
from nowcastnet.utils.logging import setup_logging, log_configs
from nowcastnet.utils.visualizing import crop_frames, plot_frames
from nowcastnet.datasets.factory import dataset_provider

from utils import setup_parser, load_config, construct_configs


def inference(sess: ort.InferenceSession, dataloader: DataLoader, configs):
    logging.info("Inference started")

    np.random.seed(configs.seed)

    input_name = ort_session.get_inputs()[0].name
    noise_name = ort_session.get_inputs()[1].name
    output_name = ort_session.get_outputs()[0].name

    results_dir = configs.results_path
    os.makedirs(results_dir, exist_ok=True)

    for batch, (observed_frames, _) in enumerate(dataloader):
        logging.info(f"Batch: {batch + 1}/{len(dataloader)}")

        observed_frames = observed_frames.numpy()
        noise = np.random.rand(
            configs.batch_size,
            configs.generator_base_channels,
            configs.image_height // 32,
            configs.image_width // 32,
        ).astype(np.float32)

        predicted_frames = ort_session.run(
            [output_name], {input_name: observed_frames, noise_name: noise}
        )[0]

        result_path = os.path.join(results_dir, str(batch))
        os.makedirs(result_path, exist_ok=True)

        if configs.case_type == "normal":
            predicted_frames = crop_frames(
                frames=predicted_frames, crop_size=configs.crop_size
            )

        plot_frames(frames=predicted_frames[0], save_dir=result_path, vmin=1, vmax=40)

        if configs.save_original_data:
            np.save(os.path.join(result_path, "frames.npy"), predicted_frames[0])

    logging.info("Inference finished")
    logging.info(f"Results saved to {results_dir}")


if __name__ == "__main__":
    parser = setup_parser("Do ONNX model Inference")
    args = parser.parse_args()

    configs = load_config(args.config_path).value
    configs = construct_configs(configs, ["common", args.case_type])
    configs.__setattr__("case_type", args.case_type)

    os.makedirs(os.path.dirname(configs.path_to_log), exist_ok=True)
    setup_logging(configs.path_to_log)
    log_configs(configs)

    dataloader = dataset_provider(configs)
    logging.info(f"DataLoader created from {configs.dataset_path}")

    ort_session = ort.InferenceSession(
        configs.graph_path, providers=["CPUExecutionProvider"]
    )
    logging.info(f"ORT inference session created from {configs.graph_path}")

    inference(ort_session, dataloader, configs)
