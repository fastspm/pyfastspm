"""Exact interpolation for interlacing"""

import logging

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial import Delaunay
from tqdm import tqdm

from ..fast_movie import FastMovie

log = logging.getLogger(__name__)


def _output_y_grid(ny, nx):
    """Returns the y grid of the data points; not corrected for probe creep

    Args:
        ny: number of pixels in y direction
        nx: number of pixels in x direction

    Returns:
        y:  y meshgrid of measured datapoints
        t1: equidistant 1D grid; contains the centers of each line in y

    """

    # equidistant grid
    t1 = np.linspace(-(ny / 2), ny / 2, ny)
    dummy_yvals = np.linspace(-(ny / 2), ny / 2, ny * nx)

    # meshgrid of measured datapoints
    y = np.reshape(dummy_yvals, (ny, nx))

    for i in range(ny):
        y[i, :] = y[i, :: (-1) ** i]

    return y, t1


def _output_x_grid(ny, nx):
    """Returns the x grid of the data points

    Args:
        ny: number of pixels in y direction
        nx: number of pixels in x direction

    Returns:
        x:  x meshgrid of measured datapoints
        t2: equidistant 1D grid

    """

    # equidistant grid
    t2 = np.linspace(-(nx / 2), nx / 2, nx)

    # hysteresis in x direction
    x2 = nx / 2.0 * np.sin(t2 * np.pi / (nx + abs(t2[0] - t2[1])))

    # meshgrid of measured datapoints
    x = np.array([x2 for j in range(ny)])

    return x, t2


def get_interpolation_matrix(
    points_to_triangulate, grid_points
):  # grid_points has to be a list of tuples!

    """
    Creates matrix containing all the relevant information for interpolating
    values measured at the same relative postion to a grid.

    The matix is constructed in a sparse format to limit memory usage.
    Construction is done in the lil sparse format, which is late converted to csr format for
    faster matrix vector dot procut.

    The matrix is of the form  (number of grid points at which to interpolate the frame) x (number of measured points within one frame).
    For this reason the number of interpolation and measured points do not have to be the same.

    Frames are interpolated within one matrix vecor dot product. For this reason the frame arrays
    need to be flattened (vectorized) before multiplication.

    Args:
        points_to_triangulate: list of tuples! represnting the points of measurement
        grid_points: list of tuples! represnting the grid points at which to interpolate

    The input formatting is slightly unusuall if you are used to numpy etc.
    Here we do not give two list, each containing values for one dimension.
    Insted we give tuples of values for each point i.e. (x,y) - See QHull documentation.

    Steps:
        1) Perform delaunay triangulation on grid of measurement positions.
        2) Find dealaunay triangles which contain new grid points.
        3) For each triangle get indices of corner poinst.
        4) At corresponding point in matrix insert barycentric coordinates.
    """

    triangulation = Delaunay(points_to_triangulate)
    triangles_containing_gridpoints = triangulation.find_simplex(grid_points)
    interpolation_matrix = lil_matrix((len(grid_points), len(points_to_triangulate)))

    for i in tqdm(
        range(len(grid_points)), desc="Building interpolation matrix", unit="lines"
    ):
        triangle_corners = triangulation.simplices[triangles_containing_gridpoints[i]]
        barycentric_coords = triangulation.transform[
            triangles_containing_gridpoints[i], :2
        ].dot(
            grid_points[i]
            - triangulation.transform[triangles_containing_gridpoints[i], 2]
        )
        barycentric_coords = np.append(
            barycentric_coords, 1 - np.sum(barycentric_coords)
        )

        if triangles_containing_gridpoints[i] == -1:
            for j in range(3):
                interpolation_matrix[i, triangle_corners[j]] = np.nan
        else:
            for j in range(3):
                interpolation_matrix[i, triangle_corners[j]] = barycentric_coords[j]

    interpolation_matrix = csr_matrix(interpolation_matrix)
    return interpolation_matrix


def interpolate(
    fast_movie: FastMovie,
    offset=0.0,
    grid=None,
    image_range=None,
    interpolation_matrix_up=None,
    interpolation_matrix_down=None,
    give_grid=False,
):
    """Interpolates the pixels in a FAST movie using the analytic positions of the probe.
    Currently only available for interlaced movies.

    Args:
        fast_movie: FastMovie object
        offset: y offset of interpolation points; defaults to 0.0
        grid: precomputed grid of the actual STM tip movement,
            this supersedes the in place calculation of Bezier curves.
        image_range: range of images to be interpolated.
        interpolation_matrix_up: precomputed interpolation matrix for up frames.
        interpolation_matrix_up: precomputed interpolation matrix for down frames. This prevents
            the matrix from beeing constructed multiple times.
        give_grid: if this option is set to True, the function returns
            the interpolation matrix directly after its construction.

    Returns:
        nothing
    """

    if fast_movie.mode != "movie":
        raise ValueError("you must first reshape your data in movie mode.")

    if image_range is None:
        image_range = fast_movie.full_image_range

    if isinstance(image_range, int):
        image_range = (image_range,)

    nx = fast_movie.data.shape[2]
    ny = fast_movie.data.shape[1]

    # meshgrids
    if "i" in fast_movie.channels:
        y, t1 = _output_y_grid(
            ny, nx
        )  # Computing only the grids which are actually need might be more efficient, but this does not seem to be a bottleneck
        x, t2 = _output_x_grid(
            ny, nx
        )  # Leaving the general structure this way to make it easier to adapt to other types of grids
    else:
        y, t1 = _output_y_grid(
            2 * ny, nx
        )  # Computing only the grids which are actually need might be more efficient, but this does not seem to be a bottleneck
        x, t2 = _output_x_grid(2 * ny, nx)  #

    # correct creep in y direction
    if grid == None:
        y_up = y
        y_down = y[:, ::-1]
    else:
        y_up, y_down = grid

    xnew, ynew = np.meshgrid(t2 * fast_movie.dist_x, t1 * fast_movie.dist_y)

    ynew += offset

    if "f" in fast_movie.channels:
        yup = ynew[0::2].copy()
        ydown = ynew[0::2].copy()
        xnew = xnew[0::2].copy()
        y_up = y_up[0::2].copy()
        y_down = y_down[1::2].copy()
        x = x[0::2].copy()
        points_up = list(zip(y_up.flatten(), x.flatten()))
        points_down = list(zip(y_down.flatten(), x.flatten()))
        grid_points = list(zip(yup.flatten(), xnew.flatten()))
        grid_points_down = list(zip(ydown.flatten(), xnew.flatten()))
    if "b" in fast_movie.channels:
        yup = ynew[1::2].copy()
        ydown = ynew[1::2].copy()
        xnew = xnew[1::2].copy()
        y_up = y_up[1::2].copy()
        y_down = y_down[0::2].copy()
        x = x[1::2].copy()
        points_up = list(zip(y_up.flatten(), x.flatten()))
        points_down = list(zip(y_down.flatten(), x.flatten()))
        grid_points = list(zip(yup.flatten(), xnew.flatten()))
        grid_points_down = list(zip(ydown.flatten(), xnew.flatten()))
    elif "i" in fast_movie.channels:
        grid_points = list(zip(ynew.flatten(), xnew.flatten()))
        points_up = list(zip(y_up.flatten(), x.flatten()))
        grid_points_down = list(zip(ynew.flatten(), xnew.flatten()))
        points_down = list(zip(y_down.flatten(), x.flatten()))

    if interpolation_matrix_up is None:
        interpolation_matrix_up = get_interpolation_matrix(points_up, grid_points)
        interpolation_matrix_down = get_interpolation_matrix(
            points_down, grid_points_down
        )

    if give_grid is True:
        return interpolation_matrix_up, interpolation_matrix_down

    fast_movie.processing_log.info(
        "Performing linear 2D grid interpolation in image range {}.".format(image_range)
    )

    for _, _, frame in tqdm(
        fast_movie.iter_frames(image_range=image_range),
        desc="Interpolation",
        unit="frames",
    ):
        if "u" and "d" in fast_movie.channels:
            if frame % 2 == 0:
                fast_movie.data[frame] = interpolation_matrix_up.dot(
                    fast_movie.data[frame].flatten()
                ).reshape(ny, nx)
            else:
                fast_movie.data[frame] = interpolation_matrix_down.dot(
                    fast_movie.data[frame].flatten()
                ).reshape(ny, nx)

        if "u" and not "d" in fast_movie.channels:
            fast_movie.data[frame] = interpolation_matrix_up.dot(
                fast_movie.data[frame].flatten()
            ).reshape(ny, nx)

        if "d" and not "u" in fast_movie.channels:
            fast_movie.data[frame] = interpolation_matrix_down.dot(
                fast_movie.data[frame].flatten()
            ).reshape(ny, nx)

    fast_movie.data = np.nan_to_num(fast_movie.data)
