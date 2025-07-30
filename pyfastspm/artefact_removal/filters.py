import logging

import numpy as np
from numpy._typing import ArrayLike
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import convolve2d
from tqdm import tqdm

from ..fast_movie import FastMovie

log = logging.getLogger(__name__)


def conv_mat(
    fast_movie: FastMovie,
    matrix: ArrayLike,
    image_range: tuple[int, int] | None = None,
) -> None:
    """Applies a convolutional filter defined by a matrix to a `FastMovie`.

    Args:
        fast_movie: `FastMovie` object
        matrix: Filter kernel
        image_range: Range of images to be interpolated

    Returns:
        None: Modifies `FastMovie.data` in-place
    """
    if fast_movie.mode != "movie":
        raise ValueError("you must first reshape your data in movie mode.")

    matrix = np.array(matrix)
    leny = int((matrix.shape[0] - 1) / 2)
    lenx = int((matrix.shape[1] - 1) / 2)

    fast_movie.processing_log.info(
        "On image range {}: Applying convolutional filter: {}".format(
            image_range, matrix
        )
    )

    for _, _, frame in tqdm(
        fast_movie.iter_frames(image_range=image_range),
        desc="Convolutional filter",
        unit="frames",
    ):
        if leny == 0:
            fast_movie.data[frame] = convolve2d(
                fast_movie.data[frame], matrix, boundary="symm"
            )[:, lenx:-lenx]
        elif lenx == 0:
            fast_movie.data[frame] = convolve2d(
                fast_movie.data[frame], matrix, boundary="symm"
            )[leny:-leny, :]
        else:
            fast_movie.data[frame] = convolve2d(
                fast_movie.data[frame], matrix, boundary="symm"
            )[leny:-leny, lenx:-lenx]


def mean_2d(fast_movie: FastMovie, pixel_width: int) -> None:
    """Applies a 2D mean filter of size (pixel_width, pixel_width) to each
        frame of a `FastMovie`

    Args:
        fast_movie: A `FastMovie` instance
        pixel_width: Size of the kernel

    Returns:
        None: Modifies `FastMovie.data` in-place
    """
    kernel_shape = (pixel_width, pixel_width)
    kernel = np.ones(kernel_shape) / (pixel_width * pixel_width)
    for i, frame in enumerate(fast_movie.data):
        fast_movie.data[i] = convolve2d(frame, kernel, mode="same")


def median_2d(fast_movie: FastMovie, pixel_width: int) -> None:
    """Applies a 2D median filter of size (pixel_width, pixel_width) to each
        frame of a `FastMovie`

    Args:
        fast_movie: A `FastMovie` instance
        pixel_width: Size of the kernel

    Returns:
        None: Modifies `FastMovie.data` in-place
    """
    size = (pixel_width, pixel_width)
    for i, frame in enumerate(fast_movie.data):
        fast_movie.data[i] = median_filter(frame, size=size)


def gaussian_2d(fast_movie: FastMovie, pixel_width: int) -> None:
    """Applies a 2D gaussian filter to each frame of a `FastMovie`

    Args:
        fast_movie: A `FastMovie` object
        pixel_width: Proportional to the size of the gaussian kernel

    Returns:
        None: Modifies `FastMovie.data` in-place
    """
    for i, frame in enumerate(fast_movie.data):
        fast_movie.data[i] = gaussian_filter(frame, pixel_width - 1, truncate=0.5)
