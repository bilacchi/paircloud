import numpy as np

import paircloud._paircloud_rs as rs


def calculate_kde(data: np.ndarray, eval_points: np.ndarray, bandwidth: float = None) -> np.ndarray:
    """
    Computes KDE using the Rust backend for performance.

    Args:
        data: 1D array of data points
        eval_points: 1D array of points where the density is evaluated
        bandwidth: Optional bandwidth value. Uses Silverman's rule if None.

    Returns:
        1D array of densities corresponding to eval_points
    """
    data = np.asarray(data, dtype=np.float64)
    eval_points = np.asarray(eval_points, dtype=np.float64)

    # drop NaNs natively in python before passing to strict rust arrays
    data = data[~np.isnan(data)]
    eval_points = eval_points[~np.isnan(eval_points)]

    return rs.calc_kde(data, eval_points, bandwidth)


def calculate_quantiles(data: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Standard quantiles
    """
    data = np.asarray(data, dtype=np.float64)
    data = data[~np.isnan(data)]
    return np.quantile(data, q)
