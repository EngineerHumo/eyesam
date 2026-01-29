from typing import Tuple

import numpy as np

from .utils import binarize_mask, inscribed_center, largest_connected_component


def compute_faz_center(faz_mask: np.ndarray) -> Tuple[int, int]:
    faz_bin = binarize_mask(faz_mask)
    faz_lcc = largest_connected_component(faz_bin)
    return inscribed_center(faz_lcc)
