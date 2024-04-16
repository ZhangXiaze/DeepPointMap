from typing import Any
import torch
import numpy as np
from functools import partial


def _single_move_to_device(x, device='cpu', detach: bool = True, non_blocking: bool = False):
    if isinstance(x, torch.Tensor) and x.device != device:
        if detach:
            return x.detach().to(device, non_blocking=non_blocking)
        else:
            return x.to(device, non_blocking=non_blocking)
    if isinstance(x, np.ndarray):
        if x.dtype == np.int16 or x.dtype == np.int32 or x.dtype == np.int64 or x.dtype == np.int128:
            dtype = torch.int64
        elif x.dtype == np.float32 or x.dtype == np.float64:
            dtype = torch.float32
        elif x.dtype == np.bool8:
            dtype = torch.bool
        else:
            raise TypeError()
        if detach:
            return torch.tensor(x, dtype=dtype, device=device).detach()
        else:
            return torch.tensor(x, dtype=dtype, device=device)
    else:
        return x


def detach_to_device(x, device, non_blocking: bool = False) -> Any:
    if x is None:
        return None
    if isinstance(x, tuple):
        return (_single_move_to_device(i, device, non_blocking=non_blocking) for i in x)
    if isinstance(x, list):
        return [_single_move_to_device(i, device, non_blocking=non_blocking) for i in x]
    else:
        return _single_move_to_device(x, device, non_blocking=non_blocking)


def move_to_device(x, device, non_blocking: bool = False):
    if x is None:
        return None
    if isinstance(x, tuple):
        return (_single_move_to_device(i, device, detach=False, non_blocking=non_blocking) for i in x)
    if isinstance(x, list):
        return [_single_move_to_device(i, device, detach=False, non_blocking=non_blocking) for i in x]
    else:
        return _single_move_to_device(x, device, detach=False, non_blocking=non_blocking)


detach_to_cpu = partial(detach_to_device, device='cpu')
