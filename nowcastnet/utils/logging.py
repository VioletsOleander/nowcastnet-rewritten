import logging
import argparse
import json


def setup_logging(log_file_path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(
                log_file_path, mode='w'),
            logging.StreamHandler()
        ])


def log_configs(args: argparse.Namespace):
    configs_dict = vars(args)
    logging.info(f'Configurations:\n{json.dumps(configs_dict, indent=4)}')
