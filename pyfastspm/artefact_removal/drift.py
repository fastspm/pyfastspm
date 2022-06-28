"""
First try at movie stabilisation with
OpenCV. Parts of the code are inspired by:
learnopencv.com/video-stabilisation-using-point-feature-matching-in-opencv
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve
from scipy.signal import correlate, medfilt
from skimage.transform import resize


class Drift:
    """
    Initialise Drift class with Fast movie instance to then
    drift correct the movie data.

    Args:
        FastmovieInstance: FastMovie object
        stepsize: integer, the difference between frames that are correlated
        corrspeed: int, the difference between two correlation windows
        show_path: Parameter, if True rare and filter drift path are plotted.
        boxcar: Parameter to decide weather boxcar filter is applied to the
            drift path or not
    """

    def __init__(
        self, FastmovieInstance, stepsize=40, corrspeed=1, show_path=False, boxcar=True
    ):
        self.data = FastmovieInstance.data
        self.file = FastmovieInstance.h5file.filename.replace(".h5", ".drift.txt")
        self.processing_log = FastmovieInstance.processing_log
        self.channels = FastmovieInstance.channels
        self.stepsize = stepsize
        self.corrspeed = corrspeed
        self.n_frames = np.shape(self.data)[0]
        self.img_width = np.shape(self.data)[2]
        self.img_height = np.shape(self.data)[1]
        self.boxcar = boxcar

        if self.img_width > self.img_height:
            self.im_size = 2 ** (int(np.log2(self.img_width)) + 1)
        else:
            self.im_size = 2 ** (int(np.log2(self.img_height)) + 1)

        self.convdims = (self.im_size * 2 - 1, self.im_size * 2 - 1)
        self.transformations = np.zeros((2, self.n_frames))
        self.integrated_trans = None
        self.show_path = show_path
        if self.stepsize is None:
            self.stepsize = int(self.n_frames / 3)

    def correct(self, mode="full", known_drift=False):
        """
        handle user input to call correct functions
        and handle internals of calcualting the drift path.

        Args:
            mode: "full" or "common" decides the method to correct
                the drift
            known_drift: False or "integrated" or "sequential", Parameter
                that decides if the drift is calculated or if a known drift
                is loaded from a drift.txt file

        Returns:
            self.correct: 2D array containing x and y coordinate of the drift
        """
        print("start drift correction")
        self.processing_log.info("Drift correction mode: {}".format(mode))
        self.processing_log.info(
            "Drift calculated with stepsize: {}".format(self.stepsize)
        )
        if not known_drift:
            self._get_drift()
            self._filter_drift()
        if known_drift == "integrated":
            self.integrated_trans = np.loadtxt(
                self.file.replace(".h5", ".drift.txt")
            ).T[0:2, :]
            self.processing_log.info("Known drift used: {}".format(known_drift))
        if known_drift == "sequential":
            self.transformations = np.loadtxt(self.file.replace(".h5", ".drift.txt")).T[
                2:4, :
            ]
            self.integrated_trans = np.cumsum(self.transformations, axis=1)
            self._write_drift()
            self.processing_log.info("Known drift used: {}".format(known_drift))
        if mode == "full":
            return self._adjust_movie_buffered(), self.integrated_trans
        if mode == "common":
            return self._adjust_movie_common(), self.integrated_trans
        print("Mode not known. Available modes are full and common.")
        inp = input("What mode do you want to use. press n to abort: ")
        if inp == "n":
            return self.data
        print('continuing with mode "' + inp + '"')
        return self.correct(mode=inp)

    def _get_drift(self):
        """
        Calculation of the drift by fft correlation.
        """
        movie = np.zeros((self.n_frames, self.im_size, self.im_size))
        hamm = np.sqrt(
            np.outer(np.hamming(self.img_height), np.hamming(self.img_width))
        )
        for i in range(self.n_frames):
            imag = self.data[i, :, :]
            imag = np.asarray(imag, dtype="float32")
            imag /= imag.std()
            imag -= imag.mean()
            imag = hamm * imag
            movie[i, :, :] = resize(
                imag, (self.im_size, self.im_size), anti_aliasing=True, order=0
            )
        for i in range(self.n_frames):
            try:
                fftd = correlate(
                    movie[self.corrspeed * i, :, :],
                    movie[self.corrspeed * i + self.stepsize, :, :],
                    method="fft",
                )
                maxind = np.argmax(fftd)
                indices = np.unravel_index(maxind, self.convdims)
                effektive_shift = np.asarray(
                    [
                        [(-(self.im_size - 1) + indices[0]) / self.stepsize],
                        [(indices[1] - (self.im_size - 1)) / self.stepsize],
                    ]
                )
                self.transformations[:, i] = effektive_shift.T
            except Exception:
                pass
        # print("last found correlation indices are {}".format(indices))

    def _filter_drift(self):
        """
        smooth and filter drift path
        """
        boxwidth = 50
        boxcar = np.ones((1, boxwidth)) / boxwidth
        boxcar = boxcar[0, :]
        self.transformations[0, :] = medfilt(self.transformations[0, :], 3)
        self.transformations[1, :] = medfilt(self.transformations[1, :], 3)
        self.integrated_trans = np.cumsum(self.transformations, axis=1)
        # linear extrapolation
        pos = np.linspace(0, self.n_frames - 1, self.n_frames)
        k1, d1 = np.polyfit(
            pos[: -self.stepsize], self.integrated_trans[0, : -self.stepsize], 1
        )
        k2, d2 = np.polyfit(
            pos[: -self.stepsize], self.integrated_trans[1, : -self.stepsize], 1
        )
        self.integrated_trans[0, -self.stepsize :] = d1 + k1 * pos[-self.stepsize :]
        self.integrated_trans[1, -self.stepsize :] = d2 + k2 * pos[-self.stepsize :]
        if self.boxcar:
            self.processing_log.info(
                "Boxcar filter used with boxsize: {}".format(boxwidth)
            )
            transformations_conv = np.zeros((2, self.n_frames))
            transformations_conv[0, :] = convolve(self.integrated_trans[0], boxcar)
            transformations_conv[1, :] = convolve(self.integrated_trans[1], boxcar)
        if self.show_path is True:
            plt.plot(self.integrated_trans[0, :], self.integrated_trans[1, :])
            plt.plot(transformations_conv[0, :], transformations_conv[1, :])
            plt.title("Drift path both raw and smoothed")
            plt.show()
        if self.boxcar:
            self.integrated_trans = transformations_conv
        self._write_drift()

    def _write_drift(self):
        """
        Writes a drift.txt file
        """
        with open(self.file, "w") as fileobject:
            fileobject.write(
                "# {0:>10}   {1:>12}  {2:>12}  {3:>12} \n".format(
                    "y integrated", "x integrated", "y sequential", "x sequential"
                )
            )
            for i in range(np.shape(self.transformations)[1]):
                fileobject.write(
                    "{0:>14}   {1:>12}  {2:>12}  {3:>12} \n".format(
                        round(self.integrated_trans[0, i], 5),
                        round(self.integrated_trans[1, i], 5),
                        round(self.transformations[0, i], 5),
                        round(self.transformations[1, i]),
                        5,
                    )
                )

    def _adjust_movie_buffered(self):
        """embed movie frames into buffered background to
        move freely according to drift path. The image ration
        is changed back for interlace movies (2:1) to fit the
        overall system architecture"""
        maxy, maxx = np.max(self.integrated_trans, 1)
        miny, minx = np.min(self.integrated_trans, 1)
        buffy = int(np.round(np.abs(maxy) + np.abs(miny))) + 1
        # print("Buffer values are {} in x and {} in y.".format(buffx, buffy))
        ## This is to see effect of scaling

        if "i" in self.channels:
            self.rescale_width = int(self.im_size / 2)
            maxx = maxx / 2
            minx = minx / 2
        else:
            self.rescale_width = self.im_size

        buffx = int(np.round(np.abs(maxx) + np.abs(minx))) + 1

        corr_movie = np.zeros(
            (self.n_frames, self.im_size + int(buffy), self.rescale_width + int(buffx))
        )
        for i in range(self.n_frames):
            shift1, shift2 = self.integrated_trans[:, i]
            shift1 = int(np.round(shift1))

            if "i" in self.channels:
                shift2 = int(np.round(shift2) / 2)
            else:
                shift2 = int(np.round(shift2))
            # possibly there is a +1 in the i for the frame to be taken.
            corr_movie[
                i,
                int(abs(miny))
                + 1
                + shift1 : int(abs(miny))
                + 1
                + self.im_size
                + shift1,
                int(abs(minx))
                + 1
                + shift2 : int(abs(minx))
                + 1
                + self.rescale_width
                + shift2,
            ] = resize(
                self.data[i, :, :],
                (self.im_size, self.rescale_width),
                anti_aliasing=True,
                order=3,
            )

        print("drift correction finished")
        return corr_movie

    def _adjust_movie_common(self):
        """cut out section from movie frames, which stays constant during
        the entire movie."""
        maxy, maxx = np.max(self.integrated_trans, 1)
        miny, minx = np.min(self.integrated_trans, 1)
        buffy = int(np.round(np.abs(maxy) + np.abs(miny))) + 1
        # print(buffx, buffy)
        ## This is to see effect of scaling

        if "i" in self.channels:
            self.rescale_width = int(self.im_size / 2)
            maxx = maxx / 2
            minx = minx / 2
        else:
            self.rescale_width = self.im_size

        buffx = int(np.round(np.abs(maxx) + np.abs(minx))) + 1

        corr_movie = np.zeros(
            (self.n_frames, self.im_size - int(buffy), self.rescale_width - int(buffx))
        )
        for i in range(self.n_frames):
            shift1, shift2 = self.integrated_trans[:, -i]
            shift1 = int(np.round(shift1))

            if "i" in self.channels:
                shift2 = int(np.round(shift2) / 2)
            else:
                shift2 = int(np.round(shift2))
            # possibly there is a +1 in the i for the frame to be taken.

            corr_movie[i, :, :] = resize(
                self.data[i, :, :],
                (self.im_size, self.rescale_width),
                anti_aliasing=True,
                order=4,
            )[
                int(abs(miny))
                + 1
                + shift1 : int(abs(miny))
                + 1
                + self.im_size
                - int(buffy)
                + shift1,
                int(abs(minx))
                + 1
                + shift2 : int(abs(minx))
                + 1
                + self.rescale_width
                - int(buffx)
                + shift2,
            ]

        print("drift correction finished")
        return corr_movie


def meanfilter(data, kernel=3):
    """
    possible meanfilter.

    Args:
        data: 1D array
        kernel: Size of values that are filtered

    Returns:
        filtered: adjusted array
    """
    filtered = np.zeros(len(data))
    if kernel % 2 == 0:
        kernel += 1
    for i in range(len(data)):
        down = i - int(kernel / 2)
        up = i + int(kernel / 2) + 1
        if down < 0:
            down = 0
        if up > len(data):
            up = int(len(data))
        filtered[i] = np.mean(data[down:up])
    return filtered
