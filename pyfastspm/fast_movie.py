"""The main FastMovie class that represents a FAST movie,
with all the necessary attributes and methods."""

import logging
from pathlib import Path

import h5py as h5
import numpy as np
from packaging.version import parse
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate as corr
from tqdm import tqdm

from pyfastspm import __version__
from pyfastspm.tools.exporter import FFMPEG_VideoWriter, gsf_writer, image_writer
from pyfastspm.tools.frame_artists import get_contrast_limits, gray_to_rgb, label_image

log = logging.getLogger(__name__)

MOVIE_CHANNEL_DESCRIPTORS = ("uf", "ub", "df", "db", "udf", "udb", "ui", "di", "udi")

FRAME_CHANNEL_DESCRIPTORS = ("uf", "ub", "df", "db", "ui", "di")


class FastMovie:
    """This is the main class that represents a FAST movie

    Args:
        file_name: the name of the .h5 file to be opened
        x_phase: the x-phase shift to be applied. Defaults to the value in the metadata
        y_phase: the x-phase shift to be applied. Defaults to 1 if the metadata is 0, otherwise to the
            value in the metadata
        log_processing (boolean): if True, logs all registered data processing
            operations to a ``.log`` file in the same directory of the ``h5`` file.

    Attributes:
        data: the FAST data as a 1darray or 2darray
        metadata: all the metadata in the .h5 file as a dictionary
        default_color_map:
        default_contrast:
        full_image_range:
        channels:
        mode:
        fps:
        x_phase:
        y_phase:

    """

    def __init__(self, file_name, x_phase=None, y_phase=None, log_processing=True):
        # get file absolute path and base name for later use
        self._absolute_path = str(Path(file_name).resolve().parent)
        self._file_base_name = str(Path(file_name).stem)

        # initialize data processing logger
        self._log_file = str(Path(file_name).with_suffix(".log"))
        self.processing_log = logging.getLogger(file_name)
        log_formatter = logging.Formatter(
            "%(levelname)1.1s[%(asctime)s - %(module)s.%(funcName)s]:  %(message)s"
        )

        if log_processing:
            file_handler = logging.FileHandler(filename=self._log_file, mode="w")
        else:
            file_handler = logging.NullHandler()
        self.processing_log.addHandler(file_handler)
        file_handler.setFormatter(log_formatter)
        self.processing_log.propagate = False

        # log pyfastspm version as a header
        self.processing_log.info("using pyfastspm version %s", __version__)

        self.h5file = h5.File(file_name, mode="r")
        log.info("file " + file_name + " successfully opened.")

        # load metadata from the HDF file
        self.metadata = {}
        for key in self.h5file["data"].attrs.keys():
            self.metadata[key] = self.h5file["data"].attrs[key]

        try:
            self.metadata["Acquisition.X_Phase"] = self.metadata.pop(
                "Acquisiton.X_Phase"
            )
            self.metadata["Acquisition.Y_Phase"] = self.metadata.pop(
                "Acquisiton.Y_Phase"
            )
        except KeyError:
            pass

        # remove misspelled key, if present, and add the correct one
        try:
            self.metadata["Acquisition.LogAmp"] = self.metadata.pop(
                "Acquisition.LogAmp."
            )
        except KeyError:
            pass

        self.expected_num_images = int(self.metadata["Acquisition.NumImages"])
        self.true_num_images = self._get_correct_num_images()
        self.num_images = self.true_num_images
        self.num_frames = self.true_num_images * 4
        # if self.expected_num_images != self.true_num_images:
        #     log.warning(
        #         "true number of images differs from Acquisition.NumImages by %d",
        #         np.abs(self.expected_num_images - self.true_num_images),
        #     )

        del self.true_num_images
        del self.expected_num_images

        log.info("number of images: %d", self.metadata["Acquisition.NumImages"])

        self.data = self.h5file["data"][()].astype(
            np.float32
        )  # just initialize self.data

        # call this to set x_phase and y_phase
        self.reload_timeseries(x_phase=x_phase, y_phase=y_phase)

        # defaults for colorizing/processing output images/frames
        # default matplotlib color_map when viewing/exporting frames/movies
        self.default_color_map = "hot"
        self.default_contrast = (0.0, 1.0)  # use 100% contrast as a default

        # TODO(Carlo): the following is useless at the moment, to be
        # implemented if interesting
        self.logarithmize = not self.metadata["Acquisition.LogAmp"]

        # the following variables control the shape, channels, fps and images
        # currently stored in self.data they are reasonably initialized here
        self.channels = "timeseries"
        self.channel_list = ("timeseries",)
        self.mode = "timeseries"
        self.full_image_range = (0, self.num_images - 1)
        self.fps = None
        self.scaling = (4.0, 4.0)

        # scaling correction
        self.dist_x = 1.0
        self.dist_y = 1.0

    def close(self):
        """Closes the h5file

        Returns: nothing

        """
        filename = self.h5file.filename
        self.h5file.close()
        self.processing_log.info("h5 file closed.")
        for handler in self.processing_log.handlers:
            self.processing_log.removeHandler(handler)
            handler.flush()
            handler.close()
        log.info("file " + filename + " succesfully closed.")

    def reload_timeseries(self, x_phase=None, y_phase=None):
        """Reloads the original timeseries from the h5file.
        This is useful to reset possible unwanted changes in the original data.

        Args:
            x_phase: the desired x phase shift. Defaults to the value in the file metadata.
            y_phase: the desired y phase shift. Defaults to the value in the file metadata.

        Returns: nothing

        """

        if x_phase is None:
            self.x_phase = self.metadata["Acquisition.X_Phase"]
        else:
            self.x_phase = x_phase
        if y_phase is None:
            self.y_phase = self.metadata["Acquisition.Y_Phase"]
        else:
            self.y_phase = y_phase

        y_phase_roll = self.y_phase * self.metadata["Scanner.X_Points"] * 2
        self.data = np.roll(
            np.array(self.h5file["data"], dtype=np.float32), self.x_phase + y_phase_roll
        )

        # Correct data inversion
        inversion_factor = 1
        if self.metadata["ExperimentInfo.FileFormatVersion"] != "":
            if (
                parse(self.metadata["ExperimentInfo.FileFormatVersion"]) >= parse("2.4")
                and not self.metadata["Acquisition.LogAmp"]
            ):
                if (
                    self.metadata["GapVoltage.Invert_Data"]
                    and self.metadata["GapVoltage.Voltage"] < 0.05
                ):
                    inversion_factor *= -1
                if self.metadata["Data.Invert"]:
                    inversion_factor *= -1
                self.data *= inversion_factor

        self.mode = "timeseries"
        self.channels = "timeseries"
        self.channel_list = ("timeseries",)
        self.fps = None
        log.info(
            "loaded timeseries (x_phase = %d, y_phase = %d)", self.x_phase, self.y_phase
        )
        self.processing_log.info(
            "loaded timeseries (x_phase = %d, y_phase = %d)", self.x_phase, self.y_phase
        )
        if self.metadata["ExperimentInfo.FileFormatVersion"] != "":
            if parse(self.metadata["ExperimentInfo.FileFormatVersion"]) < parse("1.0"):
                log.warning(
                    "file format versions before 1.0 have inconsistent "
                    "x and y phase definitions. Please double-check that "
                    "the phase values are indeed the ones you want."
                )

    def reshape_data(
        self, time_series, channels, x_points, y_points, num_images, num_frames
    ):
        """
        Returns a 3D numpy array from an HDF5 file containing (image number, the 4 channels, rows).

        Args:
            time_series (1darray): the FAST data in timeseries format
            channels: a string specifying the channels to extract
            x_points (int): the number of x points
            y_points (int): the number of y points
            num_images (int): number of images
            num_frames (int): number of frames

        Returns:
            ndarray: the reshaped data as (image number, the 4 channels, rows)

        """

        data = np.reshape(time_series, (num_images, y_points * 4, x_points))

        if channels == "udf":
            data = data[:, 0 : (4 * y_points) : 2, :]
            data = np.resize(data, (num_images * 2, y_points, x_points))
            # flip every up frame upside down
            data[0 : num_frames * 2 - 1 : 2, :, :] = data[
                0 : num_frames * 2 - 1 : 2, ::-1, :
            ]

        elif channels == "udb":
            data = data[:, 1 : (4 * y_points) : 2, :]
            data = np.resize(data, (num_images * 2, y_points, x_points))
            # flip every up frame upside down
            data[0 : num_frames * 2 - 1 : 2, :, :] = data[
                0 : num_frames * 2 - 1 : 2, ::-1, :
            ]
            # flip backwards frames horizontally
            data[0 : num_frames * 2, :, :] = data[0 : num_frames * 2, :, ::-1]

        elif channels == "uf":
            data = data[:, 0 : (2 * y_points) : 2, :]
            # flip every up frame upside down
            data[0:num_frames, :, :] = data[0:num_frames, ::-1, :]
        elif channels == "ub":
            data = data[:, 1 : (2 * y_points) : 2, :]
            # flip backwards frames horizontally
            data[0:num_frames, :, :] = data[0:num_frames, :, ::-1]
            # flip every up frame upside down
            data[0:num_frames, :, :] = data[0:num_frames, ::-1, :]

        elif channels == "df":
            data = data[:, (2 * y_points) : (4 * y_points) : 2, :]
        elif channels == "db":
            data = data[:, (2 * y_points + 1) : (4 * y_points) : 2, :]
            # flip backwards frames horizontally
            data[0:num_frames, :, :] = data[0:num_frames, :, ::-1]

        elif channels == "udi":
            data = np.resize(data, (num_images * 2, y_points * 2, x_points))
            # flip backwards lines horizontally
            data[:, 1 : y_points * 2 : 2, :] = data[:, 1 : y_points * 2 : 2, ::-1]
            # flip every up frame upside down
            data[0 : num_frames * 2 - 1 : 2, :, :] = data[
                0 : num_frames * 2 - 1 : 2, ::-1, :
            ]

        elif channels == "ui":
            data = data[:, : (2 * y_points), :]
            # flip backwards lines horizontally
            data[:, 1 : y_points * 2 : 2, :] = data[:, 1 : y_points * 2 : 2, ::-1]
            # flip every up frame upside down
            data[0:num_frames, :, :] = data[0:num_frames, ::-1, :]
        elif channels == "di":
            data = data[:, (2 * y_points) :, :]
            # flip backwards lines horizontally
            data[:, 1 : y_points * 2 : 2, :] = data[:, 1 : y_points * 2 : 2, ::-1]

        else:
            raise ValueError(
                "ERROR: "
                + channels
                + " is an unsupported combination of channels in the mask"
            )
        log.info("Reshaped timeseries to movie extracting channels " + channels)
        return data

    def reshape_to_movie(self, channels="udf"):
        """Reshapes the data from a timeseries to a movie mode,
        as an array of 2D arrays with the selected channels

        Args:
            channels: (optional, string) the channels to extract.s
                Defaults to 'udf'.

        Returns: nothing

        """
        if self.mode != "timeseries":
            raise ValueError(
                "data must be in timeseries mode,"
                "probably you already reshaped your movie. "
                "If you changed your mind, use the reload_timeseries method."
            )
        if channels not in MOVIE_CHANNEL_DESCRIPTORS:
            raise ValueError(
                "'" + channels + "' is an invalid movie channel descriptor"
            )
        self.data = self.reshape_data(
            self.data,
            channels,
            x_points=self.metadata["Scanner.X_Points"],
            y_points=self.metadata["Scanner.Y_Points"],
            num_frames=self.num_frames,
            num_images=self.num_images,
        )
        self.mode = "movie"  # change the self.mode status

        if "ud" in channels:
            self.fps = self.metadata["Scanner.Y_Frequency"] * 2.0
        elif "u" in channels or "d" in channels:
            self.fps = self.metadata["Scanner.Y_Frequency"]

        self.channels = channels

        self.channel_list = self._list_channels()

        self.processing_log.info("reshaped to movie with channels " + channels)

    def frame(self, image, channel="uf", logarithmize=None):
        """Picks a frame from a movie as numpy array, specifying image number and channel

        Args:
            image: (integer) index of the image containing the frame of interest
            channel: (optional, string) channel to extract within the selected image. Defaults to 'uf'.
            logarithmize: (optional, boolean) feature not yet implemented

        Returns: a 2darray with the requested frame
        """

        if self.mode != "movie":
            raise ValueError("you must first reshape your data in movie mode.")
        if channel not in FRAME_CHANNEL_DESCRIPTORS:
            raise ValueError("'" + channel + "' is an invalid frame channel descriptor")
        if channel not in self.channel_list:
            raise ValueError(
                "'"
                + channel
                + "' "
                + "channel is not contained in '"
                + self.channels
                + "' movie"
            )
        if not self._is_valid_image_range(image, channel):
            raise ValueError(
                "there is no image {0} in the movie, the allowed "
                "range of image indexes is {1}".format(image, self.full_image_range)
            )

        if logarithmize is None:
            # logarithmize = self.logarithmize

            # we leave it like this for the moment until logarithmize
            # is implemented to avoid tons of warnings when exporting
            logarithmize = False

        if logarithmize:
            log.warning("logarithmize not yet implemented!")
            # TODO(): TO BE IMPLEMENTED!

            return self.data[self._get_frame_index(image, channel), :, :]
        else:
            return self.data[self._get_frame_index(image, channel), :, :]

    def export_frame(
        self,
        images=None,
        file_format="png",
        channel="uf",
        color_map=None,
        contrast=None,
        scaling=None,
        interp_order=3,
        average=False,
        auto_label=False,
        output_folder=None,
    ):
        """Handles the export of single FAST frames

        Args:
            images: int (tuple of two ints) representing the image (image range) to be exported
            file_format: (optional) the export file_format. Default is PNG.
            channel: (optional) the channel to export, defaults to up
            color_map: (optional) matplotlib colormap, defaults to 'hot'
            contrast: (optional) tuple of two ints, tuple of two floats between 0 and 1,
                float between 0 and 1 (see below)
            scaling: a float controlling the size scaling to be applied to the movie array.
                Defaults to 4.0
                OR
                a 2-tuple of floats controlling the size scaling to be applied to the movie array in y and x direction.
            interp_order: an integer in the 0-5 range indicating the interpolation order of the scaling.
                For more information see the `scipy.nd.zoom documentation
                <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html#scipy.ndimage.zoom>`_
            average: (optional) a boolean controlling whether to export single frames or
                the average of frames
            auto_label: (optional) controls whether a the image is labelled
                at the top left corner with its unique id
            output_folder: (optional) the output folder for the exported frames;
                defaults to the same folder of the raw data file

        Returns:
            nothing

        Notes:
            The image contrast is controlled as in the following examples:

                * (int, int): manually cuts the histogram at the specified values
                * (float, float): automatically cuts the histogram within the specified percentiles
                    at the top and at the bottom of the range, respectively
                * float: automatically cuts the histogram by including the specified fraction
                    of the histogram

        Example:

            * (-452, 4543): cuts the histogram between the specified values
            * (0.01, 0.95): removes 1% of the histogram at the bottom and 5% of the histogram
              at the top of the value range
            * 0.95: removes 2.5% of the histogram symmetrically at both ends of the value range
        """

        if self.mode != "movie":
            raise ValueError("you must first reshape your data in movie mode")

        if channel not in self.channel_list:
            raise ValueError(
                "'"
                + channel
                + "' "
                + "channel is not contained in '"
                + self.channels
                + "' movie"
            )
        log.info("exporting channel: " + channel)

        if images is None:
            images = self.full_image_range
        elif not self._is_valid_image_range(images, channel):
            raise ValueError("invalid image range specification")

        if scaling is None:
            scaling = self.scaling
        if type(scaling) in [tuple, list, np.ndarray]:
            if "i" in self.channels:
                scaling = (scaling[0] / 2, scaling[1])
        elif type(scaling) in [float, int]:
            if "i" in self.channels:
                scaling = (scaling / 2, scaling)
            else:
                scaling = (scaling, scaling)

        if auto_label and file_format == "gsf":
            log.warning("cannot label frames exported in gsf format")

        if file_format not in ["gsf", "png", "jpg", "bmp"]:
            raise ValueError("'" + file_format + "' is an unsupported file format")
        image_range = np.array(images, ndmin=1)

        if average and image_range.shape[0] == 1:
            log.warning(
                "frame averaging was requested but a single frame was specified: "
                "I will turn off averaging and continue."
            )
            average = False

        # set some defaults
        if average:
            log.info("exporting average of images: " + str(images))
        else:
            log.info("exporting image(s): " + str(images))

        if file_format.lower() != "gsf":
            if color_map is None:
                color_map = self.default_color_map
            log.info("using color_map: " + color_map)
            if contrast is None:
                contrast = self.default_contrast
                log.info("image contrast cut between " + str(contrast))

        if output_folder is None:
            output_folder = self._absolute_path

        if average:
            frame_sum = np.zeros(self.data.shape[1:3])
        for image_id, channel_id, _ in self.iter_frames(image_range):
            if channel_id == channel:
                if average:
                    frame_sum += self.frame(image_id, channel)
                else:
                    file_name = (
                        self._file_base_name
                        + "_"
                        + str(image_id)
                        + channel_id
                        + "."
                        + file_format.lower()
                    )
                    file_name_with_path = str(Path(output_folder, file_name))
                    if file_format.lower() == "gsf":
                        self.metadata["Title"] = Path(file_name).stem
                        gsf_writer(
                            self.frame(image_id, channel),
                            file_name_with_path,
                            metadata=self.metadata,
                        )
                        del self.metadata["Title"]
                    elif file_format.lower() in ["png", "jpg", "bmp"]:
                        if auto_label:
                            label_frame_id = Path(file_name).stem
                            label_frame_start_time = "{:.3f}s".format(
                                image_id / self.fps
                            )
                            text = {
                                label_frame_id: "top-left",
                                label_frame_start_time: "top-right",
                            }
                        else:
                            text = None
                        image_writer(
                            self.frame(image_id, channel),
                            file_name_with_path,
                            color_map=color_map,
                            contrast=contrast,
                            scaling=scaling,
                            interp_order=interp_order,
                            text=text,
                        )

        if average:
            file_name = (
                self._file_base_name
                + "_"
                + "["
                + str(image_range[0])
                + "-"
                + str(image_range[-1])
                + "]"
                + channel
                + "."
                + file_format.lower()
            )
            file_name_with_path = str(Path(output_folder, file_name))
            num_images = image_range[-1] - image_range[0] + 1
            frame_average = frame_sum / num_images
            if file_format.lower() == "gsf":
                self.metadata["Title"] = Path(file_name).stem
                gsf_writer(frame_average, file_name_with_path, metadata=self.metadata)
                del self.metadata["Title"]
            elif file_format.lower() in ["png", "jpg", "bmp"]:
                if auto_label:
                    label_frame_id = Path(file_name).stem
                    text = {label_frame_id: "top-left"}
                else:
                    text = None
                image_writer(
                    frame_average,
                    file_name_with_path,
                    color_map=color_map,
                    contrast=contrast,
                    scaling=scaling,
                    interp_order=interp_order,
                    text=text,
                )

    def export_movie(
        self,
        image_range=None,
        color_map=None,
        contrast=None,
        scaling=4.0,
        interp_order=3,
        auto_label=True,
        fps_factor=1.0,
        output_folder=None,
    ):
        """Exports a FAST movie to a playable video.

        Args:
            image_range: a tuple of ints controlling the image range to export
            color_map: a ``matplotlib`` colormap, default is ``hot``
            contrast: the contrast limits that will be applied to the histogram of the **whole**
                movie
            scaling: a float controlling the size scaling to be applied to the movie array.
                Defaults to 4.0
                OR
                a 2-tuple of floats controlling the size scaling to be applied to the movie array in y and x direction.
            interp_order: an integer in the 0-5 range indicating the interpolation order of the scaling.
                For more information see the `scipy.nd.zoom documentation
                <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html#scipy.ndimage.zoom>`_
            auto_label: a boolean indicating whether the frames should be labelled with frame
                numbers and timestamps
            fps_factor: a float controlling the time stretching/compression of the exported movie
            output_folder: a string representing the desired output folder. Defaults to the
                folder where the original HDF5 file resides

        Returns:
            nothing.

        """

        if type(scaling) in [tuple, list, np.ndarray]:
            if "i" in self.channels:
                scaling = (scaling[0] / 2 * self.dist_y, scaling[1] * self.dist_x)
            else:
                scaling = (scaling[0] * self.dist_y, scaling[1] * self.dist_x)
        elif type(scaling) in [float, int]:
            if "i" in self.channels:
                scaling = (scaling / 2 * self.dist_y, scaling / 2 * self.dist_x)
            else:
                scaling = (scaling * self.dist_y, scaling * self.dist_x)

        if self.mode != "movie":
            raise ValueError("you must first reshape your data in movie mode.")

        if image_range is None:
            image_range = self.full_image_range

        # check that the requested image_range is valid
        for chan in self.channel_list:
            if not self._is_valid_image_range(image_range, chan):
                raise ValueError("invalid image range specification")

        log.info("exporting image(s): " + str(image_range))

        if color_map is None:
            color_map = self.default_color_map
        log.info("using color map: " + color_map)

        contrast = get_contrast_limits(
            self.data[slice(*self._image_to_frame_range(image_range)), :, :],
            contrast=contrast,
        )
        log.info("movie contrast cut between " + str(contrast))

        if output_folder is None:
            output_folder = self._absolute_path

        extension = ".mp4"
        log.info("using default MPEG4 output container")
        file_name = (
            self._file_base_name
            + "_"
            + str(image_range[0])
            + "-"
            + str(image_range[1])
            + "_"
            + self.channels
            + extension
        )
        log.info("output file: " + file_name)
        file_name_with_path = str(Path(output_folder, file_name))

        # round the size to the closest even number to avoid issues with h264 codecs
        size_x = np.round(self.data.shape[2] * scaling[1] / 2.0).astype(int) * 2
        size_y = np.round(self.data.shape[1] * scaling[0] / 2.0).astype(int) * 2
        writer = FFMPEG_VideoWriter(
            file_name_with_path, (size_x, size_y), self.fps * fps_factor
        )

        scaling = (size_y / self.data.shape[1], size_x / self.data.shape[2])
        log.info("effective movie scaling: " + str(np.round(scaling, 3)))

        progress_bar = tqdm(
            list(self.iter_frames(image_range)), desc="Video export", unit="frames"
        )
        if auto_label:
            for image_id, channel_id, frame_id in self.iter_frames(image_range):
                frame = gray_to_rgb(
                    self.data[frame_id, :, :],
                    color_map=color_map,
                    contrast=contrast,
                    scaling=scaling,
                )
                label_frame_id = "{:g}".format(image_id) + channel_id
                label_frame_start_time = "{:.3f}s".format(frame_id / self.fps)
                frame = label_image(
                    frame,
                    text={
                        label_frame_id: "top-left",
                        label_frame_start_time: "top-right",
                    },
                    font_size=0.05,
                    border=0.01,
                )
                writer.write_frame(frame)
                progress_bar.update(1)
        else:
            for image_id, channel_id, frame_id in self.iter_frames(image_range):
                frame = gray_to_rgb(
                    self.data[frame_id, :, :],
                    color_map=color_map,
                    contrast=contrast,
                    scaling=scaling,
                )
                writer.write_frame(frame)
                progress_bar.update(1)
        writer.close()
        progress_bar.close()

        self.processing_log.info(
            "movie export completed (scaling = %s, contrast = %s, color_map = %s, fps_factor = %d)",
            scaling,
            contrast,
            color_map,
            fps_factor,
        )

    def _get_correct_num_images(self):
        """Calculates the correct total number of images in the .h5 file

        Returns:
            int: the correct total number of images
        """
        if not isinstance(self.h5file, h5.File):
            raise Exception(
                "h5file is not an instance of h5py.File: did you open the HDF5 file?"
            )

        x_points = self.h5file["data"].attrs["Scanner.X_Points"]
        y_points = self.h5file["data"].attrs["Scanner.Y_Points"]
        num_images = int(self.h5file["data"].shape[0] / (x_points * y_points * 4))

        return num_images

    def _is_valid_image_range(self, image_range, channel, strict_range_check=False):
        is_ok = False
        image_range = np.array(image_range, ndmin=1)
        if image_range.shape[0] == 1:
            frame_index = self._get_frame_index(image_range[0], channel)
            is_ok = frame_index < self.data.shape[0] and not strict_range_check
        if image_range.shape[0] == 2:
            start_index = self._get_frame_index(image_range[0], channel)
            end_index = self._get_frame_index(image_range[1], channel)
            is_ok = 0 <= start_index < end_index < self.data.shape[0]
        return is_ok

    def _list_channels(self):
        # Checking here for sanity of movie channels descriptors must be unnecessary:
        # we should have taken care of this well before this point
        #
        # if movie_channels not in MOVIE_CHANNEL_DESCRIPTORS:
        #     raise ValueError(movie_channels + ' is a malformed channel descriptor')
        if len(self.channels) == 3:
            channel_list = (
                self.channels[0] + self.channels[-1],
                self.channels[1] + self.channels[-1],
            )
        elif len(self.channels) == 2:
            channel_list = (self.channels,)
        return channel_list

    def _get_frame_index(self, image, frame_channel):
        # Checking here for sanity of frame channels descriptors must be unnecessary:
        # we should have taken care of this well before this point
        #
        # if frame_channel not in self._list_channels(self.channels):
        #     raise ValueError('channel \'' + str(frame_channel) +
        #                      '\' is not present in \'' + self.channels + '\' movie')
        if "ud" in self.channels:
            if "u" in frame_channel:
                index = image * 2
            if "d" in frame_channel:
                index = image * 2 + 1
        else:
            index = image
        return index

    def iter_frames(self, image_range=None):
        """

        Args:
            image_range: a tuple with the start and end image_id

        Yields:
            a tuple containing image index, channel identifier and frame index
        """

        if image_range is None:
            image_range = np.array(self.full_image_range, ndmin=1)
        else:
            image_range = np.array(image_range, ndmin=1)
        if "ud" in self.channels:
            frame_id = image_range[0] * 2
        else:
            frame_id = image_range[0]
        for image_id in range(image_range[0], image_range[-1] + 1):
            for channel_id in self.channel_list:
                yield image_id, channel_id, frame_id
                frame_id += 1

    def _image_to_frame_range(self, image_range):
        start_frame = self._get_frame_index(image_range[0], self.channel_list[0])
        end_frame = self._get_frame_index(image_range[1], self.channel_list[-1])
        return start_frame, end_frame

    def correct_phase(
        self, index_frame_to_correlate, sigma_gauss=0, manual_x=0, manual_y=0
    ):

        if self.mode != "movie":
            self.reshape_to_movie("udi")

        # -4 to disregard the upper and lower most two rows
        if index_frame_to_correlate is None:
            xphase_autocorrection = 0
        else:
            num_of_correlated_lines = (len(self.data[0, :, 0]) - 4) / 2
            correlation_peak_values = np.zeros(int(num_of_correlated_lines))

            frame_to_correlate = self.data[index_frame_to_correlate]

            frame_to_correlate -= frame_to_correlate.mean()
            frame_to_correlate /= frame_to_correlate.std()

            create_hamming = np.outer(
                np.ones(len(self.data[0, :, 0])), np.hamming(len(self.data[0, 0, :]))
            )
            frame_to_correlate = frame_to_correlate * create_hamming

            if sigma_gauss != 0:
                frame_to_correlate[::2] = gaussian_filter(
                    frame_to_correlate[::2], sigma_gauss
                )
                frame_to_correlate[1::2] = gaussian_filter(
                    frame_to_correlate[1::2], sigma_gauss
                )

            for i in range(2, len(self.data[0, :, 0]) - 2, 2):
                # create foreward different mean - like finite difference approx in numerical differentiation
                correlational_data_forewards = corr(
                    frame_to_correlate[i, :], frame_to_correlate[i + 1, :]
                )
                correlational_data_backwards = corr(
                    frame_to_correlate[i, :], frame_to_correlate[i - 1, :]
                )
                max_val = (
                    np.argmax(correlational_data_forewards)
                    + np.argmax(correlational_data_backwards)
                ) / 2
                correlation_peak_values[int(i / 2 - 1)] = max_val

            mean_correlation_peak_value = np.mean(correlation_peak_values)
            raw_xphase_correction = (
                mean_correlation_peak_value - (len(self.data[0, 0, :]) - 1)
            ) / 2  # -1 to get correct index
            xphase_autocorrection = int(np.round(raw_xphase_correction))

            log.info(
                "Automatic xphase detection yielded a raw value of {} which was rounded to {}".format(
                    round(raw_xphase_correction, 3), xphase_autocorrection
                )
            )

        self.reload_timeseries(
            y_phase=manual_y,
            x_phase=self.metadata["Acquisition.X_Phase"]
            + xphase_autocorrection
            + manual_x,
        )

        x_phase = (
            +xphase_autocorrection + self.metadata["Acquisition.X_Phase"] + manual_x
        )

        return x_phase
