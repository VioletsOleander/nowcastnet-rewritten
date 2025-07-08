"""Compare the evaluation result of ONNX model with Torch model."""

import os

import matplotlib.pyplot as plt
import numpy as np

from utils import setup_parser, load_config, construct_configs


def plot_lines(x,  y1, y2, title, x_label, y_label, x_ticks, y_ticks, save_dir) -> None:
    fig = plt.figure()
    ax = plt.axes()

    ax.plot(x, y1, label='ONNX')
    ax.plot(x, y2, label='Torch')

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.legend()

    plt.savefig(os.path.join(save_dir, title + '.png'), dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = setup_parser(
        'Compare the evaluation result of ONNX model with Torch model')
    args = parser.parse_args()

    configs = load_config(args.config_path).value
    configs = {key: value + args.case_type
               for key, value in configs.items()}
    configs = construct_configs(configs)

    os.makedirs(configs.save_dir, exist_ok=True)

    print('Comparing ONNX and Torch evaluation results')
    onnx_numpy_files = sorted([filename for filename in os.listdir(configs.onnx_eval_results_path)
                               if filename.endswith('.npy')])
    torch_numpy_files = sorted([filename for filename in os.listdir(configs.torch_eval_results_path)
                                if filename.endswith('.npy')])

    for onnx_numpy_file, torch_numpy_file in zip(onnx_numpy_files, torch_numpy_files):
        print('Comparing', onnx_numpy_file, torch_numpy_file)
        onnx_numpy = np.load(os.path.join(
            configs.onnx_eval_results_path, onnx_numpy_file))
        torch_numpy = np.load(os.path.join(
            configs.torch_eval_results_path, torch_numpy_file))

        title = onnx_numpy_file.split('.')[0].upper()
        if 'CSI' in title:
            y_label = 'CSI'
        else:
            y_label = 'CSIN'

        plot_lines(x=np.arange(0, len(onnx_numpy)),
                   y1=onnx_numpy,
                   y2=torch_numpy,
                   title=title,
                   x_label='Prediction Interval (10 min)',
                   y_label=y_label,
                   x_ticks=range(0, len(onnx_numpy), 3),
                   y_ticks=np.arange(0, 1, 0.2),
                   save_dir=configs.save_dir)
        print('Comparison saved to', configs.save_dir)
    print('Comparison finished')
