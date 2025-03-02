import argparse
import logging
import os

from nowcastnet.datasets.factory import dataset_provider
from nowcastnet.utils.parsing import setup_parser
from nowcastnet.utils.logging import setup_logging, log_configs
from nowcastnet.utils.preprocessing import preprocess
from nowcastnet.utils.visualizing import crop_frames
from nowcastnet.evaluation.metrics import compute_csi, compute_csi_neighbor, compute_psd

import numpy as np
from matplotlib.colors import Normalize


def refine_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # positional arguments
    parser.add_argument('results_path', type=str,
                        help='path of the inference results')

    # evaluation configuration arguments
    evaluation_group = parser.add_argument_group(
        'evaluation configuration arguments')
    evaluation_group.add_argument('--csi_threshold', type=int, default=16,
                                  help='precipitation rate threshold for CSI calculation')
    evaluation_group.add_argument('--pooling_kernel_size', type=int, default=2,
                                  help='kernel size of maxpooling in CSI_neighbor calculation')

    # other configuration arguments
    other_group = parser.add_argument_group('other configuration arguments')
    other_group.add_argument('--preprocessed', action='store_true',
                             help='whether the dataset is preprocessed, if not, the dataset will be preprocessed')
    other_group.add_argument('--path_to_preprocessed', type=str,
                             help='path to store the preprocessed dataset, only used when preprocessed is False')
    other_group.add_argument('--save_original_data', type=bool, default=True,
                             help='whether to save the preprocessed original numpy ndarray data')
    other_group.add_argument('--path_to_log', type=str, default='evaluate.log',
                             help='path to store the log file')

    return parser


def prepare_configs(configs: argparse.Namespace) -> argparse.Namespace:
    configs.pred_length = configs.total_length - configs.input_length

    return configs


def evaluate(configs: argparse.Namespace):
    logging.info('Evaluation started')

    predicted_sample_dirs = os.listdir(configs.results_path)
    truth_sample_dirs = os.listdir(configs.dataset_path)
    sample_dirs = zip(predicted_sample_dirs, truth_sample_dirs)

    for sample_idx, (predicted_sample_dir, truth_sample_dir) in enumerate(sample_dirs):
        logging.info(f'Sample: {sample_idx+1}/{len(predicted_sample_dirs)}')

        # load predicted and truth frames
        predicted_sample_dir = os.path.join(
            configs.results_path, predicted_sample_dir)
        truth_sample_dir = os.path.join(
            configs.dataset_path, truth_sample_dir)

        predicted_frames = np.load(os.path.join(
            predicted_sample_dir, 'frames.npy'))
        truth_frames = np.load(os.path.join(
            truth_sample_dir, 'future', 'frames.npy'))

        # cropping
        predicted_frames = crop_frames(frames=predicted_frames,
                                       crop_size=configs.crop_size)
        truth_frames = crop_frames(frames=truth_frames,
                                   crop_size=configs.crop_size)

        # normalization
        norm = Normalize(vmin=0, vmax=40)
        predicted_frames = norm(predicted_frames)
        truth_frames = norm(truth_frames)
        frames = zip(predicted_frames, truth_frames)

        # evaluation
        for frame_idx, (predicted_frame, truth_frame) in enumerate(frames):
            csi = compute_csi(
                predicted_frame,
                truth_frame,
                threshold=norm(configs.csi_threshold))
            csi_neighbor = compute_csi_neighbor(
                predicted_frame,
                truth_frame,
                threshold=norm(configs.csi_threshold),
                kernel_size=configs.pooling_kernel_size)
            psd_pred, freq_pred = compute_psd(predicted_frame)
            psd_truth, freq_truth = compute_psd(truth_frame)
            mean_psd_diff = np.abs(psd_pred - psd_truth).mean()

            message = f'Frame: {frame_idx+1:0>2d}/{len(predicted_frames)} '
            message += f'(Time: {10 * (frame_idx+1):>3d}/{10 * len(predicted_frames)} min) '
            message += f'CSI: {csi:.5f}, CSIN: {csi_neighbor:.5f}, '
            message += f'Mean PSD diff: {mean_psd_diff:.5f}'
            # message += '\n'.join([f'PSD_truth - PSD_pred: {truth - pred:.5f} (freq {freq})'
            #  for truth, pred, freq in zip(psd_truth, psd_pred, freq_pred)])
            logging.info(message)


if __name__ == "__main__":
    parser = refine_parser(setup_parser(
        description='Run NowcastNet evaluation or dataset preprocessing'))
    args = parser.parse_args()
    configs = prepare_configs(args)

    setup_logging(configs.path_to_log)
    log_configs(configs)

    if not configs.preprocessed:
        dataloader = dataset_provider(configs)
        logging.info(f'DataLoader created from {configs.dataset_path}')

        preprocess(dataloader, configs)
        configs.dataset_path = configs.path_to_preprocessed

    evaluate(configs)
