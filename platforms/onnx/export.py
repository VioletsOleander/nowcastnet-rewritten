"""Export NowcastNet from torch.nn.Module to ONNX graph."""

import torch
from torch.export.dynamic_shapes import Dim
from nowcastnet.model.nowcastnet import NowcastNet

from utils import setup_parser, load_config, construct_configs


def refine_parser(parser):
    parser.add_argument(
        "--output_name",
        type=str,
        default="nowcastnet",
        help="Name of the output ONNX graph",
    )
    parser.add_argument(
        "--dynamic", action="store_true", help="Enable dynamic shape exporting"
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="artifacts",
        help="Directory to save the artifacts, like reports and the serialized exported program",
    )

    return parser


def prepare_kwargs(configs, args):
    kwargs = {
        "input_names": ["observed_frames", "noise"],
        "output_names": ["predicted_frames"],
        "dynamo": True,
        "external_data": False,
        "report": True,
        "optimize": True,
        "verify": True,
        "profile": True,
        "dump_exported_program": True,
        "artifacts_dir": args.artifacts_dir,
    }
    if not args.dynamic:
        kwargs["f"] = (
            f"{args.artifacts_dir}/{args.output_name}_{configs.image_height}.onnx"
        )
    else:
        kwargs["f"] = f"{args.artifacts_dir}/{args.output_name}_dynamic.onnx"

        batch_dim = Dim("batch_size")
        kwargs["dynamic_shapes"] = {
            "observed_frames": [
                batch_dim,
                Dim.STATIC,
                Dim("image_height"),
                Dim("image_width"),
            ],
            "noise": [batch_dim, Dim.STATIC, Dim("noise_height"), Dim("noise_width")],
        }
    return kwargs


if __name__ == "__main__":
    parser = refine_parser(
        setup_parser("Export NowcastNet from torch.nn.Module to ONNX graph")
    )
    args = parser.parse_args()

    configs = load_config(args.config_path).value
    configs = construct_configs(configs, ["common", args.case_type])

    print("Loading model")
    model = NowcastNet(configs)
    model.load_state_dict(torch.load(configs.weights_path))
    model.to(configs.device)
    model.eval()
    print("Model loaded")

    print("Generating random input and noise")
    random_input = torch.randn(
        configs.batch_size,
        configs.input_length,
        configs.image_height,
        configs.image_width,
    ).to(configs.device)
    random_noise = torch.randn(
        configs.batch_size,
        configs.generator_base_channels,
        configs.image_height // 32,
        configs.image_width // 32,
    ).to(configs.device)
    print("Random input and noise generated")

    print("Exporting model to ONNX")
    kwargs = prepare_kwargs(configs, args)
    torch.onnx.export(model=model, args=(random_input, random_noise), **kwargs)
    print(f"Model exported to ONNX and saved to {kwargs['f']}")
