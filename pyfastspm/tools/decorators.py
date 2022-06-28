"""
Contains all the decorators
"""
import numpy as np


def frame_serializer(func):
    def proc(data, **kwargs):
        if data.ndim == 2:
            return func(data, **kwargs)
        elif data.ndim == 3:
            return np.asarray(
                [func(data[i, :, :], **kwargs) for i in range(data.shape[0])]
            )

    return proc
