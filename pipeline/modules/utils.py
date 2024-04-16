import colorlog as logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import math
import torch
import open3d as o3d
from torch import Tensor
from typing import Dict, List, Tuple
from system.modules.utils import PoseTool


class Recorder:
    def __init__(self):
        self.record_dict: Dict[str, List] = {}
        self.reduction_func = {
            'min': self.min,
            'max': self.max,
            'mean': self.mean,
            'best': self.best,
            'none': lambda x: x
        }

    def add_dict(self, metric_dict: dict):
        for key, value in metric_dict.items():
            if key not in self.record_dict.keys():
                self.record_dict[key] = []
            self.record_dict.get(key).append(value)

    def add_item(self, key: str, value):
        if key not in self.record_dict.keys():
            self.record_dict[key] = []
        self.record_dict.get(key).append(value)

    def mean(self) -> dict:
        return_dict = {}
        for key, value in self.record_dict.items():
            if len(value) > 0:
                return_dict[key] = sum(value) / len(value)
        return return_dict

    def max(self) -> dict:
        return_dict = {}
        for key, value in self.record_dict.items():
            if len(value) > 0:
                return_dict[key] = max(value)
        return return_dict

    def min(self) -> dict:
        return_dict = {}
        for key, value in self.record_dict.items():
            if len(value) > 0:
                return_dict[key] = min(value)
        return return_dict

    def best(self) -> dict:
        return_dict = {}
        for key, value in self.record_dict.items():
            if len(value) > 0:
                if value[0] > value[-1]:
                    return_dict[key] = min(value)
                else:
                    return_dict[key] = max(value)
        return return_dict

    def tostring(self, reduction='best') -> str:
        assert reduction in ['min', 'max', 'mean', 'best', 'none']
        reduction_dic = self.reduction_func.get(reduction)()
        string = ''
        if len(reduction_dic) > 0:
            for key, value in reduction_dic.items():
                if isinstance(value, list):
                    value_str = value
                else:
                    value_str = f'{value:4.5f}'
                string += f'\t{key:<20s}: ({value_str})\n'
            string = '\n' + string
        return string

    def clear(self):
        self.record_dict.clear()


class Optimizer:
    def __init__(self, args):
        self.name = args.type.lower()
        self.kwargs = args.kwargs
        if self.name == 'adamw':
            self.optimizer = torch.optim.AdamW
        elif self.name == 'adam':
            self.optimizer = torch.optim.Adam
        elif self.name == 'sgd':
            self.optimizer = torch.optim.SGD
        else:
            raise NotImplementedError

    def __call__(self, parameters):
        return self.optimizer(params=parameters, **self.kwargs)


class IdentityScheduler(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def step(self):
        pass


class Scheduler:
    def __init__(self, args):
        self.name = args.type.lower()
        self.kwargs = args.kwargs
        if self.name == 'identity':
            self.scheduler = IdentityScheduler
        elif self.name == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
        elif self.name == 'cosine_restart':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        else:
            raise NotImplementedError

    def __call__(self, optimizer):
        return self.scheduler(optimizer=optimizer, **self.kwargs)


class fakecast:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def try_load_state_dict(model, state_dict, name='model', log=True):
    model_keys = model.state_dict().keys()
    file_keys = state_dict.keys()
    if model_keys == file_keys:
        try:
            model.load_state_dict(state_dict)
            if log:
                logger.info(f"{name} loaded successfully.")
            return
        except:
            if log:
                logger.warning(f"{name} loaded failed.")
            return
    else:
        missing = model_keys - file_keys
        warnings_str = f'{name} loaded with {len(model_keys)} in model, {len(file_keys)} in file.\n'
        if len(missing) != 0:
            warnings_str += f"{len(missing)} missing parameters (in model):\n" + ", ".join(missing) + '\n'
        unexpected = file_keys - model_keys
        if len(unexpected) != 0:
            warnings_str += f"{len(unexpected)} unexpected parameters (in file):\n" + ", ".join(
                unexpected) + '\n'
        try:
            model.load_state_dict(state_dict, strict=False)
            if log:
                logger.warning(warnings_str)
            return
        except:
            if log:
                logger.warning(f"{name} loaded failed.")
            return
