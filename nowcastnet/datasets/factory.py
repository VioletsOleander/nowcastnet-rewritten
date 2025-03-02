from nowcastnet.datasets.radar_dataset import RadarDataset

from argparse import Namespace
from torch.utils.data import DataLoader


def dataset_provider(configs: Namespace) -> DataLoader:
    if configs.dataset_name == 'radar':
        dataset_config = {}
        dataset_config['image_height'] = configs.image_height
        dataset_config['image_width'] = configs.image_width
        dataset_config['total_length'] = configs.total_length
        dataset_config['input_length'] = configs.input_length
        dataset_config['dataset_path'] = configs.dataset_path
        dataset_config['input_data_type'] = 'float32'
        dataset_config['output_data_type'] = 'float32'

        dataset = RadarDataset(dataset_config)
    else:
        raise ValueError(f'unkonwn dataset: {configs.dataset_name}')

    dataloader = DataLoader(dataset=dataset,
                            batch_size=configs.batch_size,
                            shuffle=False,
                            drop_last=True,
                            num_workers=configs.cpu_workers)

    return dataloader
