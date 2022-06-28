"""
This module contains tools for exporing.
It contains a class for movie export with ffmpeg
as well as tools for single FAST frame export.
"""

import logging
import os
import subprocess as sp
from subprocess import DEVNULL

from PIL import Image

from .frame_artists import gray_to_rgb, label_image

log = logging.getLogger(__name__)


def try_cmd(cmd):
    try:
        popen_params = {"stdout": sp.PIPE, "stderr": sp.PIPE, "stdin": DEVNULL}

        # This was added so that no extra unwanted window opens on windows
        # when the child process is created
        if os.name == "nt":
            popen_params["creationflags"] = 0x08000000

        proc = sp.Popen(cmd, **popen_params)
        proc.communicate()
    except Exception as err:
        return False, err
    else:
        return True, None


if os.name == "nt":
    FFMPEG_BIN = "ffmpeg.exe"
else:
    FFMPEG_BIN = "ffmpeg"

if not try_cmd(FFMPEG_BIN)[0]:
    log.warning("ffmpeg is unavailable on your system: movie export will NOT work")


class FFMPEG_VideoWriter:
    """A class for FFMPEG-based video writing.

    A class to write videos using ffmpeg. ffmpeg will write in a large
    choice of formats.

    Args:
        file_name: any filename like 'video.mp4' etc. but if you want to avoid
          complications it is recommended to use the generic extension
          '.avi' for all your videos.
        size: size (width,height) of the output video in pixels.
        fps: Frames per second in the output video file.
        codec: FFMPEG codec. It seems that in terms of quality the hierarchy is
          'rawvideo' = 'png' > 'mpeg4' > 'libx264'
          'png' manages the same lossless quality as 'rawvideo' but yields
          smaller files. Type ``ffmpeg -codecs`` in a terminal to get a list
          of accepted codecs.
          Note for default 'libx264': by default the pixel format yuv420p
          is used. If the video dimensions are not both even (e.g. 720x405)
          another pixel format is used, and this can cause problem in some
          video readers.
        preset: sets the time that FFMPEG will take to compress the video. The slower,
          the better the compression rate. Possibilities are: ultrafast,superfast,
          veryfast, faster, fast, medium (default), slow, slower, veryslow,
          placebo.
        bitrate: only relevant for codecs which accept a bitrate. "5000k" offers
          nice results in general.

    """

    def __init__(
        self,
        file_name,
        size,
        fps,
        codec="libx264",
        preset="medium",
        bitrate=None,
        logfile=None,
        threads=None,
        ffmpeg_params=None,
    ):

        if not try_cmd(FFMPEG_BIN)[0]:
            raise OSError(
                "Cannot find ffmpeg executable. "
                "Please make it available and re-import pyfastspm"
            )

        if logfile is None:
            logfile = sp.PIPE

        self.filename = file_name
        self.codec = codec
        self.ext = self.filename.split(".")[-1]

        # order is important
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-loglevel",
            "error" if logfile == sp.PIPE else "info",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            "%dx%d" % (size[0], size[1]),
            "-pix_fmt",
            "rgb24",
            "-r",
            "%.02f" % fps,
            "-i",
            "-",
            "-an",
        ]
        cmd.extend(["-vcodec", codec, "-preset", preset])

        if ffmpeg_params is not None:
            cmd.extend(ffmpeg_params)

        # TODO: allow more flexible bitrates for libx264 codec

        if bitrate is not None:
            cmd.extend(["-b", bitrate])
        elif codec == "libx264":
            cmd.extend(["-b", "5000k"])

        if threads is not None:
            cmd.extend(["-threads", str(threads)])
        else:
            cmd.extend(["-threads", str(0)])

        if (codec == "libx264") and (size[0] % 2 == 0) and (size[1] % 2 == 0):
            cmd.extend(["-pix_fmt", "yuv420p"])
        else:
            log.warning(
                "movies exported with odd sizes ({0:g}x{1:g}) "
                "will not be played by QuickTime. "
                "In case you need such player, please chose a different scaling. "
                "Otherwise, use VLC.\n".format(size[0], size[1])
            )
        cmd.extend([file_name])

        popen_params = {"stdout": DEVNULL, "stderr": logfile, "stdin": sp.PIPE}

        # This was added so that no extra unwanted window opens on windows
        # when the child process is created
        if os.name == "nt":
            popen_params["creationflags"] = 0x08000000

        self.proc = sp.Popen(cmd, **popen_params)

    def write_frame(self, img_array):
        """Writes one frame in the file."""
        try:
            self.proc.stdin.write(img_array.tostring())
        except IOError as err:
            ffmpeg_error = str(self.proc.communicate()[1])
            error = str(err) + (
                "\n\nFFMPEG encountered "
                "the following error while writing file %s:"
                "\n\n %s" % (self.filename, ffmpeg_error)
            )

            if "Unknown encoder" in ffmpeg_error:

                error += (
                    "\n\nThe video export "
                    "failed because FFMPEG didn't find the specified "
                    "codec for video encoding (%s). Please install "
                    "this codec or change the codec when calling "
                    "write_videofile. For instance:\n"
                    "  >>> clip.write_videofile('myvid.webm', codec='libvpx')"
                ) % (self.codec)

            elif "incorrect codec parameters ?" in ffmpeg_error:

                error += (
                    "\n\nThe video export "
                    "failed, possibly because the codec specified for "
                    "the video (%s) is not compatible with the given "
                    "extension (%s). Please specify a valid 'codec' "
                    "argument in write_videofile. This would be 'libx264' "
                    "or 'mpeg4' for mp4, 'libtheora' for ogv, 'libvpx for webm. "
                    "Another possible reason is that the audio codec was not "
                    "compatible with the video codec. For instance the video "
                    "extensions 'ogv' and 'webm' only allow 'libvorbis' (default) as a"
                    "video codec."
                ) % (self.codec, self.ext)

            elif "encoder setup failed" in ffmpeg_error:

                error += (
                    "\n\nThe video export "
                    "failed, possibly because the bitrate you specified "
                    "was too high or too low for the video codec."
                )

            elif "Invalid encoder type" in ffmpeg_error:

                error += (
                    "\n\nThe video export failed because the codec "
                    "or file extension you provided is not a video"
                )

            raise IOError(error)

    def close(self):
        self.proc.stdin.close()
        if self.proc.stderr is not None:
            self.proc.stderr.close()
        self.proc.wait()

        del self.proc


#### Frame export


def image_writer(
    data, file_name, color_map, contrast, scaling=4, interp_order=3, text=None
):
    """Writes a gives 2D array to an image file

    Writes a given 2D array to an image file.

    Args:
        data: a 2D array with the frame data
        file_name: a string representing the output file name
        color_map: a valid ``matplotlib`` colormap
        contrast: the contrast specified as in ``tools.color_mapper.get_contrast_limits``
        scaling: a float representing the output scaling factor
        text: the text to be superimposed on the image, as in ``tools.frame_artists.label_image``

    Returns:
        nothing

    """
    rgb_data = gray_to_rgb(
        data,
        color_map=color_map,
        contrast=contrast,
        scaling=scaling,
        interp_order=interp_order,
    )
    if text is not None:
        rgb_data = label_image(rgb_data, text=text, font_size=0.04, border=0.01)
    img = Image.fromarray(rgb_data).convert("RGB")
    img.save(file_name)
    log.info("successfully written " + file_name)


def gsf_writer(data, file_name, metadata=None):
    """Write a 2D array to a Gwyddion Simple Field 1.0 file format

    Args:
        file_name: the name of the output (any extension will be replaced)
        data: an arbitrary sized 2D array of arbitrary numeric type
        metadata: (optional) a dictionary containing additional metadata to be included in the file

    Returns:
        nothing
    """

    x_res = data.shape[1]
    y_res = data.shape[0]

    data = data.astype("float32")

    if file_name.rpartition(".")[1] == ".":
        file_name = file_name[0 : file_name.rfind(".")]

    gsf_file = open(file_name + ".gsf", "wb")

    # prepare the metadata
    if metadata is None:
        metadata = {}
    metadata_string = ""
    metadata_string += "Gwyddion Simple Field 1.0" + "\n"
    metadata_string += "XRes = {0:d}".format(x_res) + "\n"
    metadata_string += "YRes = {0:d}".format(y_res) + "\n"
    for i in metadata.keys():
        try:
            metadata_string += i + " = " + "{0:G}".format(metadata[i]) + "\n"
        except:
            metadata_string += i + " = " + str(metadata[i]) + "\n"

    gsf_file.write(metadata_string.encode("utf-8", "surrogatepass"))
    gsf_file.write(b"\x00" * (4 - len(metadata_string) % 4))
    gsf_file.write(data.tobytes(None))
    gsf_file.close()
    log.info("Successfully wrote " + file_name + ".gsf")
