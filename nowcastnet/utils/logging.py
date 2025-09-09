import json
import logging

from nowcastnet.utils.parsing import EvaluationConfig, InferenceConfig


def setup_logging(log_file_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="w"),
            logging.StreamHandler(),
        ],
    )


def log_configs(configs: InferenceConfig | EvaluationConfig):
    configs_dict = vars(configs)
    logging.info(f"Configurations:\n{json.dumps(configs_dict, indent=4)}")
