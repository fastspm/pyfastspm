import numpy as np

from ..fast_movie import FastMovie


def align_rows(fast_movie: FastMovie, align_type: str = "mean") -> None:
    """Align rows along the slow scanning direction for each frame

    Args:
        image: 2D numpy array.
        baseline: Defines how baselines are estimated; 'mean' (default),
            'median', 'poly2', 'poly3'.

    Returns:
        None: Mutates `fast_movie.data` in-place
    """
    for i, frame in enumerate(fast_movie.data):
        fast_movie.data[i], _ = _align_img(frame, baseline=align_type, axis=1)


def level_plane(fast_movie: FastMovie):
    """Corrects for image tilting by subtraction of a plane for each frame

    Args:
        image: 2D numpy array

    Returns:
        None: Mutates `fast_movie.data` in-place
    """
    for i, frame in enumerate(fast_movie.data):
        fast_movie.data[i], _ = _plane_img(frame)


def fix_zero(fast_movie):
    """Add a constant to all the data to move the minimum to zero.

    Args:
        image: 2D numpy array.

    Returns:
        None: Mutates `fast_movie.data` in-place
    """
    for i, frame in enumerate(fast_movie.data):
        fast_movie.data[i] = _fixzero_img(frame, to_mean=False)


def _fixzero_img(image: np.ndarray, to_mean: bool = False) -> np.ndarray:
    """Add a constant to all the data to move the minimum (or the mean value) to zero.

    Args:
        image: 2D numpy array.
        to_mean: bool, optional. If true move mean value to zero, if false
            move mimimum to zero (default).

    Returns:
        numpy array.
    """

    if to_mean:
        fixed = image - image.mean()
    else:
        fixed = image - image.min()

    return fixed


def _plane_img(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Corrects for image tilting by subtraction of a plane.

    Args:
        image: 2D numpy array

    Returns:
        Flattened image as 2d numpy array and the substracted background
    """

    bkg_x = _poly_bkg(image.mean(axis=0), 1)
    bkg_y = _poly_bkg(image.mean(axis=1), 1)

    bkg_xx = np.apply_along_axis(_fill, 1, image, bkg_x)
    bkg_yy = np.apply_along_axis(_fill, 0, image, bkg_y)

    bkg = bkg_xx + bkg_yy
    planned = image - bkg

    return planned, bkg


def _align_img(
    image: np.ndarray, baseline: str = "mean", axis: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Align rows along `axis`.

    Args:
        image: 2d numpy array.
        baseline: Defines how baselines are estimated; 'mean' (default), 'median', 'poly2', 'poly3'.
        axis: Axis along wich calculate the baselines.

    Returns:
        Corrected 2d numpy array and the substracted background.
    """

    if baseline == "mean":
        bkg = np.apply_along_axis(_mean_bkg, axis, image)
    elif baseline == "median":
        bkg = np.apply_along_axis(_median_bkg, axis, image)
    elif baseline == "poly2":
        bkg = np.apply_along_axis(_poly_bkg, axis, image, 2)
    elif baseline == "poly3":
        bkg = np.apply_along_axis(_poly_bkg, axis, image, 3)

    aligned = image - bkg

    return aligned, bkg


def _mean_bkg(line):
    return np.full(line.shape[0], line.mean())


def _median_bkg(line):
    return np.full(line.shape[0], np.median(line))


def _poly_bkg(line, poly_degree):
    x = np.linspace(-0.5, 0.5, line.shape[0])
    coefs = np.polyfit(x, line, poly_degree)
    return np.polyval(coefs, x)


def _fill(line, value):
    return value
