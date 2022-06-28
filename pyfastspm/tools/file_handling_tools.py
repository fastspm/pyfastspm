"""Helper functions to implement batch operations on multiple FAST files."""

import base64
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm

from ..fast_movie import FastMovie
from .frame_artists import gray_to_rgb

log = logging.getLogger(__name__)


def preview_folder(folder="."):
    """Returns an HTML file with the preview of the movies in the given folder

    Args:
        folder [optional]: a string representing the folder where to search for the ``.h5`` files

    Returns: Nothing

    Example:

        >>> pyfastspm.preview_folder('tests/')

    will create an HTML file 'tests/preview_tests.html' with preview and basic
    parameters set of the movies in the folder
    """

    # Initialize HTML string
    html_string = "<!DOCTYPE html>\n<html>\n"
    html_string += """<head>
    <style>
    table, th, td {
    border: 1px solid black;
    border-collapse: collapse;
    }
    th, td {
    padding: 15px;
    }
    </style>\n</head>\n"""
    html_string += "<body>\n<table>\n"

    # Save images
    for file in h5_files_in_folder(folder, with_path=True):
        html_string += file_to_html(file)

    # End HTML string
    html_string += "</table>\n</body>\n</html>"

    # Write HTML string to file
    p = Path(folder)
    filepath = str(p.absolute()) + "\\preview_" + p.absolute().name + ".html"
    html_file = open(filepath, "w")
    html_file.write(html_string)
    html_file.close()

    log.info("Successfully exported to " + filepath)


def file_to_html(file):
    ft = FastMovie(file)
    preview = preview_to_html(ft)
    params = params_to_html(ft)
    file_entry = "<tr><td>{}\n</td><td>\n{}</td></tr>\n".format(preview, params)
    return file_entry


def preview_to_html(ft):
    ft.reshape_to_movie()
    quick_decos(ft, image_range=0)
    rgb_frame = gray_to_rgb(ft.frame(image=0, channel="uf"))
    img_frame = Image.fromarray(rgb_frame).convert("RGB")
    output = BytesIO()
    img_frame.save(output, format="JPEG")
    frame = output.getvalue()
    preview = '<img src="'
    preview += "data:image/jpeg;base64, " + base64.b64encode(frame).decode("utf-8")
    preview += '" alt="empty" style="height:300px;"/>'
    return preview


def params_to_html(ft):
    attrs = {
        "Sample": "ExperimentInfo.Sample",
        "Comment": "ExperimentInfo.Comment",
        "Bias": "GapVoltage.Voltage",
        "Current": "Regulator.Setpoint",
        "X pixels": "Scanner.X_Points",
        "Y pixels": "Scanner.Y_Points",
        "X amplitude": "Scanner.X_Amplitude",
        "Y amplitude": "Scanner.Y_Amplitude",
        "X phase": "Acquisition.X_Phase",
        "Y phase": "Acquisition.Y_Phase",
        "X freq": "Scanner.X_Frequency",
        "Y freq": "Scanner.Y_Frequency",
        "Z amplitude": "Scanner.Z_Amplitude",
        "Z phase": "Scanner.Z_to_X_Phase",
        "Samples/pixel": "Acquisition.SamplesPerPixel",
        "Temperature": "ExperimentInfo.Temperature",
        "Date": "ExperimentInfo.Time",
        "Software version": "ExperimentInfo.SoftwareVersion",
    }
    params = "Filename: {}\n".format(ft.h5file.filename)
    params += "<br>FPS: {:.4g}\n".format(ft.fps)
    for k, v in attrs.items():
        if isinstance(ft.metadata[v], np.float64):
            try:
                params += "<br>{} ({}): {:.4g}\n".format(
                    k, ft.metadata[v + ".Unit"].decode("utf-8"), ft.metadata[v]
                )
            except KeyError:
                params += "<br>{}: {:.4g}\n".format(k, ft.metadata[v])
        elif isinstance(ft.metadata[v], bytes):
            params += "<br>{}: {}\n".format(k, ft.metadata[v].decode("utf-8"))
        else:
            try:
                params += "<br>{} ({}): {}\n".format(
                    k, ft.metadata[v + ".Unit"].decode("utf-8"), ft.metadata[v]
                )
            except KeyError:
                params += "<br>{}: {}\n".format(k, ft.metadata[v])
    params += "<br>Pixel freq (Hz): {:.6g}\n".format(
        ft.fps * ft.metadata["Scanner.X_Points"] * ft.metadata["Scanner.Y_Points"]
    )
    return params


def h5_files_in_folder(folder, with_path=False):
    """Returns a list of the ``.h5`` files in the specified folder

    Args:
        folder: a string representing the folder where to search for the ``.h5`` files
        with_path: a boolean indicating whether the output file names should include the full path.
            Defaults to ``False``.

    Returns: a list of the ``.h5`` files in the specified folder

    Example:

        >>> for i in pyfastspm.tools.h5_files_in_folder('tests/'):
        ...     print(i)

    where you can replace ``print(i)`` with whatever operation to be repeated
    on the single ``h5`` files contained in the ``'tests/'``.
    """

    p = Path(folder)
    file_list = [str(file if with_path else file.name) for file in list(p.glob("*.h5"))]

    return sorted(file_list, key=lambda tstr: int(tstr.split("_")[-1].split(".")[0]))


def unprocessed_in_folder(folder, extension="mp4", with_path=False):
    """Returns the base names for all ``.h5`` files in the specified folder that have no sibling ``.mp4`` file

    Args:
        folder: a string representing the folder where to search for the files
        extension: the extension (without dot) of sibilings to be filter the ``.h5`` files.
            Defaults to ``mp4``.
        with_path: a boolean indicating whether the output file names should include the full path.
            Defaults to ``False``.

    Returns: a list of the unprocessed ``.h5`` files in the specified folder

    """

    p = Path(folder)

    unprocessed_file_list = []

    for file in list(p.glob("*.h5")):
        if len(list(p.glob(file.stem + "*." + extension))) == 0:
            unprocessed_file_list.append(str(file if with_path else file.name))

    return sorted(
        unprocessed_file_list, key=lambda tstr: int(tstr.split("_")[-1].split(".")[0])
    )


def quick_decos(fast_movie: FastMovie, image_range=None):
    """Corrects cosine distortion in a frame or a movie with fast re-sampling onto
    an appropriate meshgrid.

    Args:
        fast_movie: (FastMovie) an instance of the FastMovie class
        image_range: an int or a tuple indicating the image range to decos

    Returns:
        a modified version the FastMovie.data attribute

    References:
        Alexander Jussupow, *Analysis of fast-STM movies using Python* -
        Research internship report - Department of Chemistry, TU Munich (2014)

    """
    if fast_movie.mode != "movie":
        raise ValueError("you must first reshape your data in movie mode.")

    if image_range is None:
        image_range = fast_movie.full_image_range

    if isinstance(image_range, int):
        image_range = (image_range,)

    nx = fast_movie.data.shape[2]
    ny = fast_movie.data.shape[1]
    t1 = np.linspace(-(ny / 2), ny / 2, ny)
    t2 = np.linspace(-(nx / 2), nx / 2, nx)
    # hysteresis in x direction
    x2 = nx / 2.0 * np.sin(t2 * np.pi / (nx))
    for _, _, frame in tqdm(
        fast_movie.iter_frames(image_range=image_range),
        desc="quick decos",
        unit="frames",
    ):
        f2D = RectBivariateSpline(t1, x2, fast_movie.data[frame])
        fast_movie.data[frame] = f2D(t1, t2)
