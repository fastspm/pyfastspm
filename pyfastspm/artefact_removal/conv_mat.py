import logging

import numpy as np
from scipy.signal import convolve2d
from tqdm import tqdm

from ..fast_movie import FastMovie

log = logging.getLogger(__name__)


def conv_mat(fast_movie: FastMovie, matrix, image_range=None):
    """Applies a convolutional filter defined by a matrix to a FastMovie.

    Args:
        fast_movie: FastMovie object
        matrix: filter matrix
        image_range: range of images to be interpolated

    Returns:
        nothing

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
