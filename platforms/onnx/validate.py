"""Validate the precision of the ONNX model output."""

import re

from utils import setup_parser, load_config, construct_configs

import torch
import numpy as np
import onnxruntime
from nowcastnet.model.nowcastnet import NowcastNet


def refine_parser(parser):
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')

    return parser


def get_torch_output(config, numpy_input, noise):
    torch_input = torch.tensor(numpy_input)
    torch_input = torch_input.to(device=config.device)
    noise = torch.tensor(noise)
    noise = noise.to(device=config.device)

    torch_model = NowcastNet(config)
    torch_model.load_state_dict(torch.load(
        config.weights_path, weights_only=True))
    torch_model.eval()
    torch_model.to(device=config.device)

    with torch.no_grad():
        torch_output = torch_model(torch_input, noise)

    return torch_output


def get_onnx_output(config, numpy_input, noise):
    ort_session = onnxruntime.InferenceSession(
        config.graph_path, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    noise_name = ort_session.get_inputs()[1].name
    output_name = ort_session.get_outputs()[0].name

    onnx_output = ort_session.run(
        [output_name], {input_name: numpy_input, noise_name: noise})[0]

    return onnx_output


if __name__ == '__main__':
    parser = refine_parser(setup_parser(
        'Validate the precision of the ONNX model output'))
    args = parser.parse_args()

    configs = load_config(args.config_path).value
    torch_configs = construct_configs(configs, ['common', args.case_type])
    onnx_configs = construct_configs(configs, [f'onnx_{args.case_type}'])

    np.random.seed(args.seed)
    numpy_input = np.random.randn(
        torch_configs.batch_size,
        torch_configs.input_length,
        torch_configs.image_height,
        torch_configs.image_width).astype(np.float32)
    print(f'input frames shape: {numpy_input.shape}')

    noise = np.random.randn(
        torch_configs.batch_size,
        torch_configs.generator_base_channels,
        torch_configs.image_height // 32,
        torch_configs.image_width // 32).astype(np.float32)
    print(f'noise shape: {noise.shape}')

    torch_output = get_torch_output(
        torch_configs, numpy_input, noise).cpu().numpy()
    print(f'torch output shape: {torch_output.shape}')

    onnx_output = get_onnx_output(
        onnx_configs, numpy_input, noise)
    print(f'onnx output shape: {onnx_output.shape}')

    try:
        torch.testing.assert_close(torch_output, onnx_output)
        print('success!')
    except AssertionError as e:
        error_message = e.args[0]
        print(error_message)

        indices = re.findall(r'index \(([0-9, ]*)\)', error_message)

        greatest_abs_diff_index = tuple(
            [int(num) for num in indices[0].split(', ')])
        greatest_rel_diff_index = tuple(
            [int(num) for num in indices[1].split(', ')])

        message = 'elements in greatest absolute difference:\n'
        message += f'torch_output[{indices[0]}]: {torch_output[greatest_abs_diff_index]:.8f}, '
        message += f'onnx_output[{indices[0]}]: {onnx_output[greatest_abs_diff_index]:.8f}'
        print(message)

        message = 'elements in greatest relative difference:\n'
        message += f'torch_output[{indices[1]}]: {torch_output[greatest_rel_diff_index]:.8f}, '
        message += f'onnx_output[{indices[1]}]: {onnx_output[greatest_rel_diff_index]:.8f}'
        print(message)
