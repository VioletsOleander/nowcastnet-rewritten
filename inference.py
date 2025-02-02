import argparse
import logging
import json
import os

from model.nowcastnet import NowcastNet
from datasets.factory import dataset_provider

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                'inference.log', mode='w'),
            logging.StreamHandler()
        ])


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run NowcastNet inference', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # positoinal args
    parser.add_argument('weights_path', type=str,
                        help='path of the pretrained model weights')
    parser.add_argument('dataset_path', type=str,
                        help='path of the dataset')
    parser.add_argument('results_path', type=str,
                        help='path to store the generated results', nargs='?', default='results')

    # optional args
    # data information arguments
    data_info_group = parser.add_argument_group('data informatoin arguments')
    data_info_group.add_argument(
        '--dataset_name', type=str, default='radar', help='name of target dataset')
    data_info_group.add_argument('--input_length', type=int, default=9,
                                 help='number of input frames for the model')
    data_info_group.add_argument('--total_length', type=int, default=29,
                                 help='number of total input frames')
    data_info_group.add_argument('--image_height', type=int,
                                 default=512, help='height of input frames')
    data_info_group.add_argument('--image_width', type=int, default=512,
                                 help='width of input frames')

    # data loading and processing arguments
    data_process_group = parser.add_argument_group(
        'data loading and processing arguments')
    data_process_group.add_argument('--cpu_workers', type=int, default=0,
                                    help='number of working processes for data loading')
    data_process_group.add_argument('--case_type', type=str,
                                    default='normal', choices=['normal', 'large'],
                                    help='different case_type corresponds to different image processing method for generated frames')
    data_process_group.add_argument(
        '--batch_size', type=int, default=1, help='size of minibatch')

    # model configuration arguments
    model_group = parser.add_argument_group('model configuration arguments')
    model_group.add_argument('--generator_base_channels', type=int, default=32,
                             help='number of generator base channels')
    model_group.add_argument('--device', type=str, default='cpu',
                             help='device to run the model')

    return parser


def prepare_configs(args: argparse.Namespace) -> argparse.Namespace:
    configs = args
    configs.pred_length = configs.total_length - configs.input_length
    configs.gen_decoder_input_channels = configs.generator_base_channels * 10

    return configs


def log_configs(args: argparse.Namespace):
    configs_dict = vars(args)
    logging.info(f'Configurations:\n{json.dumps(configs_dict, indent=4)}')


def plot_and_save(frames, save_dir, vmin, vmax, cmap='viridis'):
    for frame_idx, frame in enumerate(frames):
        fig = plt.figure()
        ax = plt.axes()

        ax.set_axis_off()

        alpha = frame.copy()
        alpha[alpha < 1] = 0
        alpha[alpha > 1] = 1

        ax.imshow(frame, alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap)

        plt.savefig(os.path.join(save_dir, f'frame-{frame_idx}.png'))
        plt.close()


def crop_frames(frames, case_type, crop_size=192):
    batch_size, n_frames, height, width = frames.shape

    if case_type == 'normal':
        h_center = height // 2
        w_center = width // 2

        frames = frames[:, :,
                        h_center-crop_size:h_center + crop_size,
                        w_center-crop_size:w_center+crop_size]

    return frames


def inference(model: nn.Module, dataloader: DataLoader, configs: argparse.Namespace):
    logging.info('Inference started')

    results_dir = configs.results_path
    os.makedirs(results_dir, exist_ok=True)

    model.to(device=configs.device)
    model.eval()

    for batch, observed_frames in enumerate(dataloader):
        logging.info(f'Batch: {batch}/{len(dataloader)}')

        observed_frames = observed_frames.to(device=configs.device)
        with torch.no_grad():
            predicted_frames = model(observed_frames)
        predicted_frames = predicted_frames.detach().cpu().numpy()

        result_path = os.path.join(results_dir, str(batch))
        os.makedirs(result_path, exist_ok=True)

        predicted_frames = crop_frames(frames=predicted_frames,
                                       case_type=configs.case_type)

        plot_and_save(frames=predicted_frames[0],
                      save_dir=result_path,
                      vmin=1,
                      vmax=40)

    logging.info('Inference finished')


if __name__ == "__main__":
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    setup_logging()

    parser = setup_parser()
    args = parser.parse_args()
    configs = prepare_configs(args)
    log_configs(configs)

    model = NowcastNet(configs)
    model.load_state_dict(torch.load(configs.weights_path))
    logging.info(f'Model weights loaded from {configs.weights_path}')

    dataloader = dataset_provider(configs)
    logging.info(f'DataLoader created from {configs.dataset_path}')

    inference(model, dataloader, configs)
