import numpy as np
from openeo import DataCube


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("size must be odd")
    r = size // 2
    y, x = np.mgrid[-r : r + 1, -r : r + 1]
    k = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    k = k / k.sum()
    return k


def smooth_probabilities_gaussian(
    cube: DataCube, kernel_size: int = 9, sigma: float = 2.0
) -> DataCube:
    """
    Apply Gaussian smoothing to the probability data cube.

    Args:
        cube: Input DataCube containing probability values.
        kernel_size: Size of the Gaussian kernel (must be odd).
        sigma: Standard deviation for the Gaussian kernel.

    Returns:
        DataCube: Smoothed DataCube.
    """
    kernel = gaussian_kernel(size=kernel_size, sigma=sigma)

    smoothed_cube = cube.apply_kernel(
        kernel=kernel,
        # border='reflect'
    )
    return smoothed_cube
