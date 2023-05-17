import numpy as np
from scipy.spatial.distance import cdist


def get_sparsity_metrics(z_hat: np.ndarray):
    """Returns mean, max, and min sparsity of z_hat rows.

    Args:
        z_hat (np.ndarray): Numpy array of sparse vectors.
    """
    sparsity = (z_hat != 0).sum(axis=1)
    return sparsity.mean(), sparsity.max(), sparsity.min()


def get_mean_recon_error(Y: np.ndarray, Y_hat: np.ndarray):
    """Computes how well Y_hat reconstructs Y on average.

    Args:
        Y (np.ndarray): True data.
        Y_hat (np.ndarray): Estimated values.
    """
    return np.mean(np.sum((Y_hat - Y) ** 2, axis=1) / np.sum(Y ** 2, axis=1))


def get_mean_col_recovery(true_dict: np.ndarray, fitted_dict: np.ndarray, dist_type: str = "cosine"):
    """Computes average over distance from each column of true_dict to nearest neighbor of fitted_dict.

    Args:
        true_dict (np.ndarray): Ground truth dictionary.
        fitted_dict (np.ndarray): The learned dictionary.
        dist_type (str): The distance metric to use. See scipy.spatial.distance.cdist.
    """
    return np.mean(
        np.minimum(
            np.amin(cdist(true_dict, fitted_dict, metric=dist_type), axis=1),
            np.amin(cdist(true_dict, -1 * fitted_dict, metric=dist_type), axis=1),
        )
    )