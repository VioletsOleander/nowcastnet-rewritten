import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from nowcastnet.model.nowcastnet import NowcastNet
from nowcastnet.datasets.factory import dataset_provider
from nowcastnet.utils.visualizing import plot_frames, crop_frames
from nowcastnet.utils.parsing import setup_parser
from nowcastnet.utils.logging import setup_logging, log_configs


def refine_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # positional arguments
    parser.add_argument('weights_path', type=str,
                        help='path of the pretrained model weights')
    parser.add_argument('results_path', type=str,
                        help='path to store the generated results')

    # model configuration arguments
    model_group = parser.add_argument_group('model configuration arguments')
    model_group.add_argument('--generator_base_channels', type=int, default=32,
                             help='number of generator base channels')
    model_group.add_argument('--device', type=str, default='cpu',
                             help='device to run the model')

    # other configuration arguments
    other_group = parser.add_argument_group('other configuration arguments')
    other_group.add_argument('--save_original_data', type=bool, default=True,
                             help='whether to save the inferenced original numpy ndarray data')
    other_group.add_argument('--path_to_log', type=str, default='inference.log',
                             help='path to store the log file')
    other_group.add_argument('--seed', type=int, default=42,
                             help='random seed for reproducibility')

    return parser


def prepare_configs(configs: argparse.Namespace) -> argparse.Namespace:
    configs.pred_length = configs.total_length - configs.input_length
    configs.gen_decoder_input_channels = configs.generator_base_channels * 10

    return configs


def inference(model: nn.Module, dataloader: DataLoader, configs: argparse.Namespace):
    logging.info('Inference started')

    np.random.seed(configs.seed)

    results_dir = configs.results_path
    os.makedirs(results_dir, exist_ok=True)

    model.to(device=configs.device)
    model.eval()

    for batch, (observed_frames, _) in enumerate(dataloader):
        logging.info(f'Batch: {batch+1}/{len(dataloader)}')

        observed_frames = observed_frames.to(device=configs.device)
        noise = np.random.randn(
            configs.batch_size,
            configs.generator_base_channels,
            configs.image_height//32,
            configs.image_width//32).astype(np.float32)
        noise = torch.from_numpy(noise).to(device=configs.device)
        # noise = torch.randn(
        #     configs.batch_size,
        #     configs.generator_base_channels,
        #     configs.image_height//32,
        #     configs.image_width//32).to(device=configs.device)

        with torch.no_grad():
            predicted_frames = model(observed_frames, noise)
        predicted_frames = predicted_frames.detach().cpu().numpy()

        result_path = os.path.join(results_dir, str(batch))
        os.makedirs(result_path, exist_ok=True)

        if configs.case_type == 'normal':
            predicted_frames = crop_frames(frames=predicted_frames,
                                           crop_size=configs.crop_size)

        plot_frames(frames=predicted_frames[0],
                    save_dir=result_path,
                    vmin=1,
                    vmax=40)

        if configs.save_original_data:
            np.save(os.path.join(result_path, 'frames.npy'),
                    predicted_frames[0])

    logging.info('Inference finished')
    logging.info(f'Results saved to {results_dir}')


if __name__ == "__main__":
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    parser = refine_parser(setup_parser(
        description='Run NowcastNet inference'))
    args = parser.parse_args()
    configs = prepare_configs(args)

    setup_logging(configs.path_to_log)
    log_configs(configs)

    model = NowcastNet(configs)
    model.load_state_dict(torch.load(configs.weights_path))
    logging.info(f'Model weights loaded from {configs.weights_path}')

    dataloader = dataset_provider(configs)
    logging.info(f'DataLoader created from {configs.dataset_path}')

    inference(model, dataloader, configs)
