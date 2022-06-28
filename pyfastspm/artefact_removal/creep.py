import logging
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Rbf, splev, splrep
from scipy.optimize import LinearConstraint, NonlinearConstraint, curve_fit, minimize
from scipy.signal import correlate as corr

# from ..fast_movie import FastMovie

log = logging.getLogger(__name__)


class Creep:
    """
    Create a compressed grid to correct for STM creep by virtue of
    interpolating measured values of successive up and down measurements
    to minimize the difference.

    Args:
        FastMovie_instance: object of fast_movie class
        index_to_linear: float, gives the position of the transition
        from the sine function into a linear function.
        creep_mode: sets the creep function used in fit_creep(). Applicable arguments
        are 'sin' and 'root'. The Bezier creep fit has to be called separately.
    """

    def __init__(self, FastMovie_instance, index_to_linear=0.5, creep_mode="sin"):
        self.channels = FastMovie_instance.channels
        self.processing_log = FastMovie_instance.processing_log
        self.number_xpixels = FastMovie_instance.metadata["Scanner.X_Points"]
        self.number_ypixels = FastMovie_instance.metadata["Scanner.Y_Points"] * 2
        if index_to_linear >= 1:
            index_to_linear = index_to_linear / self.number_ypixels
        self.rel_ind_raw = index_to_linear
        if FastMovie_instance.mode != "movie":
            raise ValueError('The FastMovie instance must be in mode "movie"')
        else:
            self.data = FastMovie_instance.data
        self.ygridfold, self.ygridstraight = self.ygrid()
        self.pixel_shift = self._get_shift(0.5)
        creep_functions = {"sin": self.sin_one_param, "root": self.root_creep}
        creep_bounds = {
            "sin": ([0.2, np.pi / 2 - 0.2]),
            "root": ([0, -np.inf], [np.inf, 0]),
        }
        self.creep_function = creep_functions[creep_mode]
        self.bounds = creep_bounds[creep_mode]

    def _get_shift(self, imrange):
        """
        Args:
            imrange: percentage of movie that should be used to evaluate shift. 0.5 recommended

        Returns: number of pixels frames are shifted in the middle of the frame.
        """
        outer_cutoff = (1 - imrange) / 2
        if "i" in self.channels:
            out_cut_ind = int(self.number_ypixels * outer_cutoff)
        else:
            out_cut_ind = int(self.number_ypixels / 2 * outer_cutoff)
        l2 = np.zeros(self.number_xpixels)
        for i in range(self.number_xpixels):
            dat1 = self.data[3, out_cut_ind:-out_cut_ind, i]
            dat2 = self.data[4, out_cut_ind:-out_cut_ind, i]
            dat1 = (dat1 - dat1.mean()) / dat1.std()
            dat2 = (dat2 - dat2.mean()) / dat2.std()
            l1 = corr(dat1, dat2)
            l2[i] = np.argmax(l1)

        mean_shift = np.mean(l2)
        diff_in_pixels = len(dat1) - (mean_shift + 1)

        self.processing_log.info(
            "Correlation of image rows retunred a shift of {} pixels in the middle of the frame.".format(
                abs(diff_in_pixels)
            )
        )

        return abs(diff_in_pixels)

    def sin_one_param(self, phase):
        """
        Args:
            phase: phase of sin function used to approximate creep.
        Returns: Creep adapted grid of expected STM tip movement.
        """
        self.ind = int(
            (1 - self.rel_ind_raw) * self.number_xpixels * self.number_ypixels
        )
        # max_val is used to set the largest value the sin function will reach.
        max_val = 1 - self.pixel_shift / (self.number_ypixels * self.rel_ind_raw)
        premult_versine = 2 * (phase * max_val - np.sin(phase)) / (np.cos(phase) - 1)

        def lin_out(x):
            return self.ygridstraight[self.ind] + (
                self.ygridstraight[-1] - self.ygridstraight[self.ind]
            ) * (
                (1 / phase)
                * (np.sin(phase * x) - premult_versine * np.sin((phase * x) / 2) ** 2)
            )

        vlin_out = np.vectorize(lin_out)

        newvals = vlin_out(
            np.linspace(0, 1, self.number_xpixels * self.number_ypixels - self.ind)
        )

        ycreep = [*self.ygridstraight[: self.ind], *newvals]

        ycreep_up = ycreep + (self.ygridstraight[-1] - ycreep[-1]) / 2

        ycreep_up = ycreep_up + [
            (ycreep_up[0] - self.ygridstraight[0]) * t
            for t in np.linspace(-1, 1, len(ycreep_up))
        ]
        return ycreep_up

    def root_creep(self, b, t0):
        """This function is adapted from:
        J I J Choi et al 2014 J. Phys.: Condens. Matter 26 225003
        y0 + a * t + b * np.sqrt(t - t0)
        According to this paper, y0 and a are constrained by the following conditions.
        Namely at t=0 -> 0 and t=t_stop -> y_max. t0 needs to be negative."""
        t = np.linspace(0, 1, self.number_ypixels * self.number_xpixels)
        y_max = self.number_ypixels
        ycreep_up = (
            -b * (-t0) ** 0.5
            + t * (y_max + b * ((-t0) ** 0.5 - (1 - t0) ** 0.5))
            + b * np.sqrt(t - t0)
        )
        ycreep_up -= y_max / 2
        return ycreep_up

    def _shape_to_grid(self, ycreep_up):
        """Convert shape of input array array to mesh."""
        ycreep_down = ycreep_up.copy()
        ycreep_down = -ycreep_down[::-1]
        ycreep_down_mesh = np.reshape(
            ycreep_down, (self.number_ypixels, self.number_xpixels)
        )
        ycreep_down_mesh[1::2, :] = ycreep_down_mesh[1::2, ::-1]

        ycreep_up_mesh = np.reshape(
            ycreep_up, (self.number_ypixels, self.number_xpixels)
        )
        ycreep_up_mesh[1::2, :] = ycreep_up_mesh[1::2, ::-1]

        return ycreep_up_mesh, ycreep_down_mesh, (ycreep_up[0], ycreep_up[-1])

    def ygrid(self):
        """
        Returns: default uncorrected y grid.
                 We are aware that this is somewhat redundant since
                 this job is beeing done in interpolate already.
                 We will sort this out in the future.
        """
        dummy_yvals = np.linspace(
            -self.number_ypixels / 2,
            self.number_ypixels / 2,
            self.number_ypixels * self.number_xpixels,
        )
        ygrid = dummy_yvals.copy()
        ygrid = np.reshape(ygrid, (self.number_ypixels, self.number_xpixels))

        for i in range(self.number_ypixels):
            ygrid[i, :] = ygrid[i, :: (-1) ** i]

        return ygrid, dummy_yvals

    def _get_diff_from_grid(self, par, *params):
        """
        This function establisches the link between "curve_fit"
        in "fit_creep" and the actual construction of the
        creepcorrected grid. Then interpolates the values of the
        recorded from the uncorrected to the corrected grid. The
        function adjusts the fit for different movie modes.

        Parameters:
            par: tupple containing info which frame and which
                row within that frame to fit.
            params: Parameters of creep function.
                   Tupple of input values for creep function.

        Returns: Array of differences between up and down frame.
            This should approach zero as the creep corrected grid
            becomes better and better.
        """

        frame, row = par
        frame = int(frame)
        row = int(row)
        ycreep_up = self.creep_function(*params)
        comp_grid_up, comp_grid_down, startstop_lin = self._shape_to_grid(ycreep_up)
        if "i" in self.channels:
            tck_up = splrep(comp_grid_up[:, row], self.data[frame, :, row], s=0)
            tck_down = splrep(comp_grid_down[:, row], self.data[frame + 1, :, row], s=0)
            newvals_up = splev(self.ygridfold[row, :], tck_up, der=0)
            newvals_down = splev(self.ygridfold[row, :], tck_down, der=0)
        elif "b" in self.channels:
            tck_up = splrep(comp_grid_up[1::2, row], self.data[frame, :, row], s=0)
            tck_down = splrep(
                comp_grid_down[0::2, row], self.data[frame + 1, :, row], s=0
            )
            newvals_up = splev(self.ygridfold[row, 1::2], tck_up, der=0)
            newvals_down = splev(self.ygridfold[row, 1::2], tck_down, der=0)
        elif "f" in self.channels:
            tck_up = splrep(comp_grid_up[0::2, row], self.data[frame, :, row], s=0)
            tck_down = splrep(
                comp_grid_down[1::2, row], self.data[frame + 1, :, row], s=0
            )
            newvals_up = splev(self.ygridfold[row, 0::2], tck_up, der=0)
            newvals_down = splev(self.ygridfold[row, 0::2], tck_down, der=0)
        else:
            self.processing_log.info("Channel Info not found in _get_diff_from_grid...")
            raise ValueError(
                'self.channels must be "i", "f" or "b" in _get_diff_from_grid. Could not detect any of those modes'
            )

        return_val = newvals_up - newvals_down
        return return_val

    def fit_creep(self, params=(0.6,), frames=[0, 2], known_params=None):
        """
        Minimizes "_get_diff_from_grid" to obtain creep corrected
        grid of predicted STM tip movement.

        Parameters:
            params: tupple containing initial guess for input parameters
                of the creep function.
            frames: n-Tuple/list deciding which frames are used for fit.
            known_params: tuple or None. Params are only calculated if this is None.
                Otherwise reestimatin is skipped and known_params is used as optimizer.

        Returns:
            2-Tuple of creep corrected up and down grids.
        """
        if known_params == None:
            print("starting creep correction")
            for frame_number_index in range(len(frames)):
                if frames[frame_number_index] % 2 != 0:
                    frames[frame_number_index] += 1
            fitresult = np.zeros(len(params))
            count = 0
            for frame in frames:
                for row in np.linspace(
                    self.data.shape[2] * 0.25, self.data.shape[2] * 0.75, 3
                ).astype(int):
                    try:
                        popt, pcov = curve_fit(
                            self._get_diff_from_grid,
                            (frame, row),
                            np.zeros_like(self.number_ypixels),
                            params,
                            bounds=self.bounds,
                        )
                        fitresult += popt
                        count += 1
                    except Exception as e:
                        print("Something went wrong.")
                        print("Caught Exception was {}".format(e))
                        print("fit attempt failed, trying next...")
                        pass
            if count == 0:  ## Only happens if all curve_fit attempts fail.
                avg_result = np.array(params)
                self.processing_log.info(
                    "Creep fitting failed on all iterations (all frames and rows). Using input parameter values {}:".format(
                        avg_result
                    )
                )
            else:
                avg_result = np.asarray(fitresult) / np.float(count)
                print("creep fit succeeded, result: {}".format(avg_result))
                self.processing_log.info(
                    "Creep fitting returned {} as optimal parameter values.".format(
                        avg_result
                    )
                )
        else:
            avg_result = known_params
            self.processing_log.info(
                "Creep parameters known. Using as {} as parameter values.".format(
                    avg_result
                )
            )
        ycreep = self.creep_function(*avg_result)
        y_up, y_down, rest = self._shape_to_grid(ycreep)

        return (y_up, y_down)

    def _Bezier(self, y, shape, pixels, Bezier_points):
        """Corrects a given y meshgrid for probe creep, using a Bezier curve

        Args:
            y: y meshgrid
            ny: number of pixels in y direction
            shape: 3-tuple describing the shape of the creep in y direction; see wiki for further information
            pixels: excess pixels at top/bottom in up/down frames
            Bezier_points: number of grid points for the numeric creep function

        Returns:
            y_up, y_down: corrected y meshgrids in both scan directions

        """

        ny = y.shape[0]

        # different creep in up and down frames
        y_up = copy(y)
        y_down = copy(y)

        ind_up = int(
            ny * (1.0 - shape[0])
        )  # point at which the y movement becomes linear in upward frames
        ind_down = int(ny * shape[0])

        # creep is approximated by a Bezier curve
        P0_up = np.array([min(y[ind_up, :]), min(y[ind_up, :])])
        P12_up = np.array([y[-1, 0] - pixels, y[-1, 0] - pixels])
        P3_up = np.array([y[-1, 0], y[-1, 0] - pixels])

        P1_up = (1.0 - shape[1]) * P0_up + shape[1] * P12_up
        P2_up = (1.0 - shape[2]) * P12_up + shape[2] * P3_up

        # since Bezier curves have the form (x,y)(t) rather than y(x), one has to numerically find y(x) on a grid
        Bezier_points = int(Bezier_points)
        t = np.linspace(0.0, 1.0, Bezier_points)

        Bezier_up = np.array(
            [
                (1 - t) ** 3 * P0_up[0]
                + 3 * (1 - t) ** 2 * t * P1_up[0]
                + 3 * (1 - t) * t**2 * P2_up[0]
                + t**3 * P3_up[0],
                (1 - t) ** 3 * P0_up[1]
                + 3 * (1 - t) ** 2 * t * P1_up[1]
                + 3 * (1 - t) * t**2 * P2_up[1]
                + t**3 * P3_up[1],
            ]
        )

        j = 0  # initialize j only once to avoid the algorithm searching the entire Bezier curve for each y element
        for i, yi in enumerate(y_up[ind_up:, :]):
            ind = ind_up + i  # actual index of the line in the original array
            if "i" in self.channels:
                fb = (-1) ** (ind % 2)  # 1 for forward, -1 for backward
            if "f" in self.channels:
                fb = -1
            if "b" in self.channels:
                fb = 1
            for ii, yii in enumerate(
                yi[::fb]
            ):  # restore time ordering which is lost by reshaping
                while j < int(Bezier_points - 1):
                    if (
                        Bezier_up[0, j] >= yii
                    ):  # find the index at which the Bezier curve passes yii
                        break
                    j += 1
                w0 = yii - Bezier_up[0, j - 1]
                w1 = Bezier_up[0, j] - yii
                y_up[ind, ::fb][ii] = (
                    w0 * Bezier_up[1, j] + w1 * Bezier_up[1, j - 1]
                ) / (
                    w0 + w1
                )  # linear interpolation between two Bezier points

        P3_down = np.array([max(y[ind_down, :]), max(y[ind_down, :])])
        P12_down = np.array([y[0, 0] + pixels, y[0, 0] + pixels])
        P0_down = np.array([y[0, 0], y[0, 0] + pixels])

        P2_down = (1.0 - shape[1]) * P3_down + shape[1] * P12_down
        P1_down = (1.0 - shape[2]) * P12_down + shape[2] * P0_down

        Bezier_down = np.array(
            [
                (1 - t) ** 3 * P0_down[0]
                + 3 * (1 - t) ** 2 * t * P1_down[0]
                + 3 * (1 - t) * t**2 * P2_down[0]
                + t**3 * P3_down[0],
                (1 - t) ** 3 * P0_down[1]
                + 3 * (1 - t) ** 2 * t * P1_down[1]
                + 3 * (1 - t) * t**2 * P2_down[1]
                + t**3 * P3_down[1],
            ]
        )

        j = 0
        for i, yi in enumerate(y_down[:ind_down, :]):
            if "i" in self.channels:
                fb = (-1) ** (ind % 2)  # 1 for forward, -1 for backward
            if "f" in self.channels:
                fb = 1
            if "b" in self.channels:
                fb = -1
            for ii, yii in enumerate(yi[::fb]):
                while j < int(Bezier_points - 1):
                    if Bezier_down[0, j] >= yii:
                        break
                    j += 1
                w0 = yii - Bezier_down[0, j - 1]
                w1 = Bezier_down[0, j] - yii
                y_down[i, ::fb][ii] = (
                    w0 * Bezier_down[1, j] + w1 * Bezier_down[1, j - 1]
                ) / (w0 + w1)

        # shift everything, such that endpoints match
        y_down -= pixels / 2.0
        y_up += pixels / 2.0

        return y_up, y_down

    def _interpolate_col(self, col, y):
        """Spline interpolation of a single column on a given y grid

        Args:
            col: the column
            y: y grid of the data in col

        Returns:
            col_int: interpolated data on an equidistant grid of the same size as the original grid

        """

        t1 = np.linspace(min(y), max(y), len(y))

        tck = splrep(y, col, s=0)
        col_int = splev(t1, tck, der=0)

        return col_int

    def _interpolate_col_param(self, input1, shape1, shape2, shape3):
        """Given the shape parameters and pixel number for the creep correction,
        compares the interpolated data of a column in an up and a down frame

        Args:
            input1: 5-tuple containing:
                y: not creep corrected y grid of the column
                up: data of the column in the up frame
                down: data of the column in the down frame
                Bezier_points: number of grid points for the numeric creep function
                w: additional weighting of the lines at the upper and lower boundary;
                    weight function is w*y**2/max(y)**2 + 1
                pixels: pixel number for creep correction
            shape1, shape2, shape3: shape parameters for creep correction

        Returns:
            on equidistant grid: difference between interpolated up and down column times the weight function

        """
        y, up, down, Bezier_points, w, pixels = input1

        y_up, y_down = self._Bezier(
            np.reshape(y, (len(y), 1)),
            (shape1, shape2, shape3),
            pixels[0],
            Bezier_points[0],
        )

        up_int = self._interpolate_col(up, y_up)
        down_int = self._interpolate_col(down, y_down)

        weight = w[0] * y**2 / max(y) ** 2 + 1

        return np.reshape((up_int - down_int), -1) * weight

    def fit_creep_bez(
        self,
        col_inds,
        frames=[0, 2],
        w=0.0,
        shape=(0.5, 2.0 / 3.0, 1.0 / 3.0),
        Bezier_points=1000,
        known_input=None,
    ):
        """Fits the shape parameters and pixel number for creep correction to minimize
        the difference between interpolated columns in up and a down frames. The fit
        is adjusted for different movie modes.

        Args:
            fast_movie: FastMovie object
            col_inds: n-tuple of indices of the columns to fit
            frames: n-tuple of indices of the frames to fit
            w: additional weighting of the lines at the upper and lower boundary;
                weight function is w*y**2/max(y)**2 + 1
                defaults to 0
            shape: 3-tuple containing the initial guess for the shape parameters;
                defaults to (.5, 2./3., 1./3.)
            Bezier_points: number of grid points for the numeric creep function

        Returns:
            opt: 4 element array containing the fitted shape parameters and pixel number (in this order)

        """

        if known_input == None:
            print("start bezier creep correction")

            self.processing_log.info(
                "Fitting creep correction at columns {} in images {}. Initial guess: shape = ({:05.4f}, {:05.4f}, {:05.4f}).".format(
                    col_inds, frames, shape[0], shape[1], shape[2]
                )
            )
            self.processing_log.info(
                "Additional weight to top and bottom image boundary: {}. Number of evaluation points for numeric creep function: {}.".format(
                    w, Bezier_points
                )
            )

            opt_list = []

            # The following is necessary to comply with new version of scipy.
            if "i" in self.channels:
                y_len = len(self.ygridfold[:, col_inds[0]])
            else:
                y_len = len(self.ygridfold[::2, col_inds[0]])
            Bezier_points = [Bezier_points for i in range(y_len)]
            w = [w for i in range(y_len)]
            pixels = [self.pixel_shift for i in range(y_len)]

            for col_ind in col_inds:
                if "i" in self.channels:
                    y = self.ygridfold[:, col_ind]
                elif "f" in self.channels:
                    y = self.ygridfold[0::2, col_ind]
                elif "b" in self.channels:
                    y = self.ygridfold[1::2, col_ind]

                for frame in frames:
                    up = self.data[frame, :, col_ind]
                    down = self.data[frame + 1, :, col_ind]
                    opt_i, _ = curve_fit(
                        self._interpolate_col_param,
                        (y, up, down, Bezier_points, w, pixels),
                        np.zeros_like(up),
                        (shape[0], shape[1], shape[2]),
                        bounds=([0.1, 0.0, 0.0], [1.0, 1.0, 1.0]),
                    )

                    opt_list.append(opt_i)

            opt = np.mean(np.array(opt_list), 0)

            self.processing_log.info(
                "Optimized creep parameters for later use: shape = ({:05.4f}, {:05.4f}, {:05.4f})".format(
                    opt[0], opt[1], opt[2]
                )
            )
            print(
                "creep fit succeeded, result: ({:05.4f}, {:05.4f}, {:05.4f})".format(
                    opt[0], opt[1], opt[2]
                )
            )
        else:
            opt = known_input
            pixels = [self.pixel_shift]
            Bezier_points = [Bezier_points]

            self.processing_log.info(
                "known parameters used = ({}, {}, {})".format(opt[0], opt[1], opt[2])
            )

        if "i" in self.channels:
            grid_up, grid_down = self._Bezier(
                self.ygridfold, opt, pixels[0], Bezier_points[0]
            )
        elif "f" in self.channels:
            ygrid = self.ygridfold[0::2]
            up, down = self._Bezier(ygrid, opt, pixels[0], Bezier_points[0])
            grid_up = np.zeros((np.shape(up)[0] * 2, np.shape(up)[1]))
            grid_down = grid_up.copy()
            grid_up[0::2] = up[:]
            grid_down[1::2] = down[:]
        elif "b" in self.channels:
            ygrid = self.ygridfold[1::2]
            up, down = self._Bezier(ygrid, opt, pixels[0], Bezier_points[0])
            grid_up = np.zeros((np.shape(up)[0] * 2, np.shape(up)[1]))
            grid_down = grid_up.copy()
            grid_up[1::2] = up[:]
            grid_down[0::2] = down[:]

        return opt, (grid_up, grid_down)
