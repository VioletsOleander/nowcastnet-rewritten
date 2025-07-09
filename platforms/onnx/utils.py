import argparse
from types import SimpleNamespace

import tomlkit


def setup_parser(description):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('config_path', type=str,
                        help='Path to the config file')
    parser.add_argument('--case_type', type=str, default='normal',
                        choices=['normal', 'large'],
                        help='Different case_type corresponds to different image processing methods for generated frames.')

    return parser


def load_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        configs = tomlkit.load(f)

    return configs


def construct_configs(configs, keys: list | None = None):
    config_dir = {}

    if keys:
        for key in keys:
            config_dir.update(**configs[key])
    else:
        config_dir = configs

    configs = SimpleNamespace(**config_dir)

    return configs
