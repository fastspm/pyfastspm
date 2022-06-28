"""
Tests for `pyfastspm.tools` module.
"""
from pathlib import Path

import numpy as np
import pytest

from pyfastspm.tools.file_handling_tools import (
    h5_files_in_folder,
    unprocessed_in_folder,
)
from pyfastspm.tools.frame_artists import get_contrast_limits

TEST_DIR = "examples/"
TEST_FILE = "examples/20141003_24.h5"
TEST_SINGLE_FRAME = "tests/test_single_frame.npy"
TEST_MULTI_FRAME = "tests/test_multi_frame.npy"


def test_h5_file_list():
    assert len(h5_files_in_folder(TEST_DIR, with_path=False)) != 0


@pytest.mark.parametrize(
    "contrast, levels",
    [
        (None, (-68, 1002)),
        (1.0, (-68, 1002)),
        ((0.0, 1.0), (-68, 1002)),
        ((-323, 454), (-323, 454)),
        ((0.025, 0.975), (18, 334)),
        (0.95, (18, 334)),
    ],
)
def test_contrast_single(contrast, levels):
    data = np.load(TEST_SINGLE_FRAME)
    output = get_contrast_limits(data, contrast=contrast)
    assert output == levels


@pytest.mark.parametrize(
    "contrast, levels",
    [
        (None, (-102, 1056)),
        (1.0, (-102, 1056)),
        ((0.0, 1.0), (-102, 1056)),
        ((-323, 454), (-323, 454)),
        ((0.025, 0.975), (12, 340)),
        (0.95, (12, 340)),
    ],
)
def test_contrast_multi(contrast, levels):
    data = np.load(TEST_MULTI_FRAME)
    output = get_contrast_limits(data, contrast=contrast)
    assert output == levels
