"""
Contains general tools to prepare and change to optics
of frames such as colormapping and drawing into frames.
"""

import logging

import matplotlib.cm as cm
import matplotlib.colors as mplc
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pkg_resources import resource_filename
from scipy.ndimage import zoom

log = logging.getLogger(__name__)


def label_image(rgb_image, text=None, font_size=0.05, border=0.01):
    """

    Superimposes the given text on an RGB image, at a chosen position.

    Args:
        rgb_image: an RGB ndarray as the input image
        text: a dictionary where the keys are the text to be written and the values are the corresponding
            positions on the image. Accepted values for text positioning are ``top-left``,
            ``top-right``, ``bottom-left``, ``bottom-right``, ``center``
        font_size: the fraction of the image width/height to be used as font height
        border: the fraction of the image width/height to be left as border

    Returns:
        an RGB ndarray

    Examples:
        >>> image_with_labels = label_image(my_rgb_image, {'Graphene edges':'top-right', 'T = 453K':'bottom-right'})

    """
    image = Image.fromarray(rgb_image).convert("RGBA")
    txt = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt)
    font = ImageFont.truetype(
        resource_filename("pyfastspm.tools.resources.fonts", "OpenSans-Semibold.ttf"),
        int(image.size[0] * font_size),
    )
    for txt_label in text.keys():
        if text[txt_label] == "top-left":
            x_pos = image.size[0] * border
            y_pos = image.size[1] * border
        elif text[txt_label] == "top-right":
            x_pos = image.size[0] * (1 - border) - font.getsize(txt_label)[0]
            y_pos = image.size[1] * border
        elif text[txt_label] == "center":
            x_pos = 0.5 * (image.size[0] - font.getsize(txt_label)[0])
            y_pos = 0.5 * (image.size[1] - font.getsize(txt_label)[1])
        elif text[txt_label] == "bottom-left":
            x_pos = image.size[0] * border
            y_pos = image.size[1] * (1 - border) - font.getsize(txt_label)[1]
        elif text[txt_label] == "bottom-right":
            x_pos = image.size[0] * (1 - border) - font.getsize(txt_label)[0]
            y_pos = image.size[1] * (1 - border) - font.getsize(txt_label)[1]
        else:
            raise ValueError("invalid specification of text position")
        draw.text((x_pos, y_pos), txt_label, fill=(255, 255, 255, 180), font=font)
    return np.array(Image.alpha_composite(image, txt))[:, :, :3]


def gray_to_rgb(
    data, color_map="hot", contrast=None, scaling=(1.0, 1.0), interp_order=3
):
    """

    Converts a single-channel image to an RGB image by applying a ``matplotlib`` color_map
    and normalizing the contrast to a given range.

    Args:
        data: input 2darray data
        color_map: one of the standard ``matplotlib`` colormaps
        contrast: a float between 0 and 1, a sequence of two floats between 0 and 1,
            or a sequence of two integers.
        scaling: 2 element list defining the scaling in both directions to be applied to the image. Defaults to no scaling (1.0, 1.0)
        interp_order: an integer in the 0-5 range indicating the interpolation order of the scaling.
                For more information see the `scipy.nd.zoom documentation
                <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html#scipy.ndimage.zoom>`_

    Returns:
        a ndarray of RGB data. RGB values range in 0...255

    Notes:
        See the documentation of the ``get_contrast_limits`` function for more details on the ``contrast`` parameter

    """
    if data.ndim != 2:
        raise ValueError("Dimension of input array is not 2.")
    value_levels = get_contrast_limits(data, contrast=contrast)
    if any(i != 1.0 for i in scaling):
        data = zoom(data, scaling, order=interp_order)
    norm = mplc.Normalize(vmin=value_levels[0], vmax=value_levels[1])
    mappable = cm.ScalarMappable(norm=norm, cmap=color_map)
    return (mappable.to_rgba(data)[:, :, 0:3] * 255).astype(np.uint8)


def get_contrast_limits(data, contrast=None):
    """Calculates the minimum and maximum values where to cut the data range according to the specified contrast range.

    Args:
        data: a single 2darray or a 1darray of 2darrays
        contrast: (optional) tuple of two ints, tuple of two floats between 0 and 1, float between 0 and 1.
            Defaults to full contrast range.

    Returns: a tuple of two ints representing the min and max values

    Notes:
        More specifically the ``contrast`` parameter can be conveniently specified as:

            * a float between 0 and 1, which is interpreted as the fraction of the image histogram
              to be **kept** in the contrast range (default)
            * a sequence of two floats between 0 and 1, indicating the fractions of the image histogram
              to be **discarded** in the contrast, at the bottom and at the top of the image range, respectively
            * a sequence of two integers is interpreted as a *manual* setting of the contrast,
              where the values correspond
              to the minimum and maximum value of the data
    """
    contrast = np.array(contrast) if contrast is not None else None
    if contrast is None:
        p1, p2 = int(data.min()), int(data.max())
    elif np.issubclass_(contrast.dtype.type, float):
        if contrast.size == 1 and 0 <= contrast <= 1:
            limits = (100.0 * (1 - contrast) / 2, 100.0 * (1 - (1 - contrast) / 2))
        elif contrast.size == 2 and 0 <= contrast[0] <= 1 and 0 <= contrast[1] <= 1:
            limits = (100.0 * contrast[0], 100.0 * contrast[1])
        else:
            raise ValueError(
                "'contrast' must be a float, a tuple of two floats between 0 and 1, or a tuple of two ints."
            )
        p1, p2 = np.percentile(data, (limits[0], limits[1]))
        log.debug("Auto-set contrast limits are {0:g} and {1:g}".format(p1, p2))

    elif np.issubclass_(contrast.dtype.type, np.integer):
        if contrast.size != 2:
            raise ValueError("'contrast' must be a sequence of two ints (or floats).")
        if contrast[0] >= contrast[1]:
            raise ValueError(
                "Lower bound of contrast must be smaller than the higher bound."
            )
        p1, p2 = contrast[0], contrast[1]
        log.debug("Manual contrast limits are {0:g} and {1:g}".format(p1, p2))

    else:
        raise ValueError(
            "'contrast' must be a float, a tuple of two floats between 0 and 1, or a tuple of two ints."
        )
    return (int(p1), int(p2))
