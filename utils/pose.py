import numpy as np
from torch import Tensor as Tensor
from typing import Tuple


def rt_global_to_relative(center_R: Tensor, center_T: Tensor, other_R: Tensor, other_T: Tensor) -> Tuple[Tensor, Tensor]:

    relative_R = center_R.transpose(-1, -2) @ other_R
    relative_T = center_R.transpose(-1, -2) @ (other_T - center_T)
    return relative_R, relative_T


def rt_global_to_relative_np(center_R: np.ndarray, center_T: np.ndarray, other_R: np.ndarray, other_T: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:

    relative_R = np.swapaxes(center_R, -1, -2) @ other_R
    relative_T = np.swapaxes(center_R, -1, -2) @ (other_T - center_T)
    return relative_R, relative_T

