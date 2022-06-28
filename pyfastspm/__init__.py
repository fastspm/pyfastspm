import logging

from pyfastspm._version import version as __version__
from pyfastspm._version import version_tuple as __version_tuple__

from .artefact_removal.conv_mat import conv_mat
from .artefact_removal.creep import Creep
from .artefact_removal.drift import Drift
from .artefact_removal.fft import (
    convert_to_spectrum,
    convert_to_timeseries,
    filter_freq,
    filter_movie,
    filter_noise,
    show_fft,
)
from .artefact_removal.interpolate import interpolate

# convenience imports
from .fast_movie import FastMovie
from .tools.error_catcher import error_catcher
from .tools.file_handling_tools import (
    h5_files_in_folder,
    preview_folder,
    unprocessed_in_folder,
)
from .tracking.pixel_trace import pixel_trace


class NullHandler(logging.Handler):
    def emit(self, record):
        pass


logging.getLogger(__name__).addHandler(NullHandler())
__FORMAT = "%(levelname)s[%(module)s.%(funcName)s]:  %(message)s"
logging.basicConfig(level=logging.INFO, format=__FORMAT)
logging.raiseExceptions = True

logging.info("Loaded pyfastspm v" + __version__)

del NullHandler
