import argparse
import logging
import os

import numpy as np
from matplotlib.colors import Normalize

from nowcastnet.datasets.factory import dataset_provider
from nowcastnet.evaluation.metrics import compute_csi, compute_csi_neighbor, compute_psd
from nowcastnet.utils.logging import log_configs, setup_logging
from nowcastnet.utils.parsing import setup_parser
from nowcastnet.utils.preprocessing import preprocess
from nowcastnet.utils.visualizing import crop_frames, plot_line


def refine_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # positional arguments (required if config_path is not provided)
    parser.add_argument(
        "infer_results_path",
        type=str,
        nargs="?",
        default=None,
        help="path of the inference results",
    )
    parser.add_argument(
        "eval_results_path",
        type=str,
        nargs="?",
        default=None,
        help="path to store the evaluation results",
    )

    # evaluation configuration arguments (optional)
    evaluation_group = parser.add_argument_group("evaluation configuration arguments")
    evaluation_group.add_argument(
        "--csi_threshold",
        type=int,
        default=16,
        help="precipitation rate threshold for CSI calculation",
    )
    evaluation_group.add_argument(
        "--pooling_kernel_size",
        type=int,
        default=2,
        help="kernel size of maxpooling in CSI_neighbor calculation",
    )

    # other configuration arguments (optional)
    other_group = parser.add_argument_group("other configuration arguments")
    other_group.add_argument(
        "--preprocessed",
        action="store_true",
        help="whether the dataset is preprocessed, if not, the dataset will be preprocessed",
    )
    other_group.add_argument(
        "--preprocessed_dataset_path",
        type=str,
        help="path to store the preprocessed dataset, only used when preprocessed is False",
    )
    other_group.add_argument(
        "--save_original_data",
        type=bool,
        default=True,
        help="whether to save the preprocessed original numpy ndarray data of the evaluation result",
    )
    other_group.add_argument(
        "--log_path",
        type=str,
        default="evaluate.log",
        help="path to store the log file",
    )

    return parser


def prepare_configs(configs: argparse.Namespace) -> argparse.Namespace:
    configs.total_length = configs.input_length + configs.pred_length

    return configs


def freq_to_wavelength(freq):
    speed_of_light = 299792458

    wavelength = speed_of_light / freq

    return wavelength / 1000


def evaluate(configs: argparse.Namespace):
    logging.info("Evaluation started")

    os.makedirs(configs.eval_results_path, exist_ok=True)

    predicted_sample_dirs = sorted(os.listdir(configs.infer_results_path))
    truth_sample_dirs = sorted(os.listdir(configs.dataset_path))
    sample_dirs = zip(predicted_sample_dirs, truth_sample_dirs)

    frameidx_to_avgcsi = {frame_idx: [] for frame_idx in range(configs.pred_length)}
    frameidx_to_avgcsin = {frame_idx: [] for frame_idx in range(configs.pred_length)}

    for sample_idx, (predicted_sample_dir, truth_sample_dir) in enumerate(sample_dirs):
        logging.info(f"Sample: {sample_idx + 1}/{len(predicted_sample_dirs)}")

        # load predicted and truth frames
        predicted_sample_dir = os.path.join(
            configs.infer_results_path, predicted_sample_dir
        )
        truth_sample_dir = os.path.join(configs.dataset_path, truth_sample_dir)

        predicted_frames = np.load(os.path.join(predicted_sample_dir, "frames.npy"))
        truth_frames = np.load(os.path.join(truth_sample_dir, "future", "frames.npy"))

        # cropping
        predicted_frames = crop_frames(
            frames=predicted_frames, crop_size=configs.crop_size
        )
        truth_frames = crop_frames(frames=truth_frames, crop_size=configs.crop_size)

        # normalization
        norm = Normalize(vmin=1, vmax=40)
        predicted_frames = norm(predicted_frames)
        truth_frames = norm(truth_frames)
        frames = zip(predicted_frames, truth_frames)

        csi_list = []
        csin_list = []

        # evaluation
        for frame_idx, (predicted_frame, truth_frame) in enumerate(frames):
            # compute metrics
            csi = compute_csi(
                predicted_frame, truth_frame, threshold=norm(configs.csi_threshold)
            )
            csi_neighbor = compute_csi_neighbor(
                predicted_frame,
                truth_frame,
                threshold=norm(configs.csi_threshold),
                kernel_size=configs.pooling_kernel_size,
            )
            psd_pred, freq_pred = compute_psd(predicted_frame)
            psd_truth, freq_truth = compute_psd(truth_frame)
            mean_psd_diff = np.abs(psd_pred - psd_truth).mean()

            # store metrics
            csi_list.append(csi)
            csin_list.append(csi_neighbor)
            frameidx_to_avgcsi[frame_idx].append(csi)
            frameidx_to_avgcsin[frame_idx].append(csi_neighbor)

            # report metrics
            message = f"Frame: {frame_idx + 1:0>2d}/{len(predicted_frames)} "
            message += (
                f"(Time: {10 * (frame_idx + 1):>3d}/{10 * len(predicted_frames)} min) "
            )
            message += f"CSI: {csi:.5f}, CSIN: {csi_neighbor:.5f}, "
            message += f"Mean PSD diff: {mean_psd_diff:.5f}"
            # message += '\n'.join([f'PSD_truth - PSD_pred: {truth - pred:.5f} (freq {freq})'
            #  for truth, pred, freq in zip(psd_truth, psd_pred, freq_pred)])
            logging.info(message)

        plot_line(
            x=np.arange(configs.pred_length) + 1,
            y=csi_list,
            x_ticks=range(0, configs.pred_length, 3),
            y_ticks=np.arange(0, 1, 0.2),
            x_label="Prediction Interval (10 min)",
            y_label="CSI",
            title=f"Precipitation (mm/h) >= {configs.csi_threshold}",
            save_dir=configs.eval_results_path,
            image_name=f"sample{sample_idx}-csi.png",
        )
        plot_line(
            x=np.arange(configs.pred_length) + 1,
            y=csin_list,
            x_ticks=range(0, configs.pred_length, 3),
            y_ticks=np.arange(0, 1, 0.2),
            x_label="Prediction Interval (10 min)",
            y_label="CSIN",
            title=f"Precipitation (mm/h) >= {configs.csi_threshold}",
            save_dir=configs.eval_results_path,
            image_name=f"sample{sample_idx}-csin.png",
        )
        logging.info(
            f"CSI and CSIN plots for sample {sample_idx} saved to {configs.eval_results_path}"
        )

        np.save(
            os.path.join(configs.eval_results_path, f"sample{sample_idx}-csi.npy"),
            csi_list,
        )
        np.save(
            os.path.join(configs.eval_results_path, f"sample{sample_idx}-csin.npy"),
            csin_list,
        )
        logging.info(
            f"CSI and CSIN arrays for sample {sample_idx} saved to {configs.eval_results_path}"
        )

    avgcsi_list = []
    avgcsin_list = []

    # report average metrics
    logging.info("Average:")
    for frame_idx in range(configs.pred_length):
        avgcsi = np.array(frameidx_to_avgcsi[frame_idx]).mean()
        avgcsin = np.array(frameidx_to_avgcsin[frame_idx]).mean()

        avgcsi_list.append(avgcsi)
        avgcsin_list.append(avgcsin)

        message = f"Frame: {frame_idx + 1:0>2d}/{len(predicted_frames)} "
        message += (
            f"(Time: {10 * (frame_idx + 1):>3d}/{10 * len(predicted_frames)} min) "
        )
        message += f"Average CSI: {avgcsi:.5f}, Average CSIN: {avgcsin:.5f}, "

        logging.info(message)

    plot_line(
        x=np.arange(configs.pred_length) + 1,
        y=avgcsi_list,
        x_ticks=range(0, configs.pred_length, 3),
        y_ticks=np.arange(0, 1, 0.2),
        x_label="Prediction Interval (10 min)",
        y_label="Average CSI",
        title=f"Precipitation (mm/h) >= {configs.csi_threshold}",
        save_dir=configs.eval_results_path,
        image_name="avg_csi.png",
    )
    plot_line(
        x=np.arange(configs.pred_length) + 1,
        y=avgcsin_list,
        x_ticks=range(0, configs.pred_length, 3),
        y_ticks=np.arange(0, 1, 0.2),
        x_label="Prediction Interval (10 min)",
        y_label="Average CSI Neighbor",
        title=f"Precipitation (mm/h) >= {configs.csi_threshold}",
        save_dir=configs.eval_results_path,
        image_name="avg_csin.png",
    )
    logging.info(f"Average CSI and CSIN plots saved to {configs.eval_results_path}")

    np.save(os.path.join(configs.eval_results_path, "avg_csi.npy"), avgcsi_list)
    np.save(os.path.join(configs.eval_results_path, "avg_csin.npy"), avgcsin_list)
    logging.info(f"Average CSI and CSIN arrays saved to {configs.eval_results_path}")


if __name__ == "__main__":
    parser = refine_parser(
        setup_parser(description="Run NowcastNet evaluation or dataset preprocessing")
    )
    args = parser.parse_args()
    configs = prepare_configs(args)

    setup_logging(configs.path_to_log)
    log_configs(configs)

    if not configs.preprocessed:
        dataloader = dataset_provider(configs)
        logging.info(f"DataLoader created from {configs.dataset_path}")

        preprocess(dataloader, configs)
        configs.dataset_path = configs.path_to_preprocessed

    evaluate(configs)
