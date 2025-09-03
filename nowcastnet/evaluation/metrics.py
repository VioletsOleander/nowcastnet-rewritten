import numpy as np
import torch
from torch.nn import MaxPool2d
from sklearn.metrics.cluster import contingency_matrix
from pysteps.utils.spectral import rapsd


def safe_access(matrix, row, col, default=0):
    try:
        return matrix[row, col]
    except IndexError:
        return default


def compute_csi(truth_field: np.ndarray, predicted_field: np.ndarray, threshold):
    predicted_field = np.where(predicted_field >= threshold, 1, 0)
    truth_field = np.where(truth_field >= threshold, 1, 0)

    matrix = contingency_matrix(truth_field.flatten(), predicted_field.flatten())

    hits = safe_access(matrix, 1, 1)
    misses = safe_access(matrix, 1, 0)
    false_alarms = safe_access(matrix, 0, 1)

    csi = hits / (hits + misses + false_alarms + 1e-10)

    return csi


def compute_csi_neighbor(
    truth_field: np.ndarray, predicted_field: np.ndarray, threshold, kernel_size=2
):
    max_pool = MaxPool2d(kernel_size=kernel_size, stride=kernel_size // 2)

    truth_field = torch.from_numpy(truth_field)
    predicted_field = torch.from_numpy(predicted_field)

    truth_field = max_pool(truth_field.unsqueeze(0)).numpy()[0]
    predicted_field = max_pool(predicted_field.unsqueeze(0)).numpy()[0]

    return compute_csi(truth_field, predicted_field, threshold)


def compute_psd(field):
    return rapsd(field, return_freq=True)
