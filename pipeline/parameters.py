from easydict import EasyDict
from collections import Iterable
import argparse
import colorlog as logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def str_to_bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise TypeError(f'str {s} can not convert to bool.')


def update_args(args, cfg: dict):
    def subdict2edict(iterable_ob):
        for i, element in enumerate(iterable_ob):
            if isinstance(element, dict):
                iterable_ob[i] = EasyDict(element)
            elif isinstance(element, Iterable) and not isinstance(element, str):
                subdict2edict(element)

    for key, value in cfg.items():
        if not hasattr(args, key):
            logger.warning(f'Found unknown parameter in yaml file: {key}')
        if isinstance(value, dict):
            value = EasyDict(value)
        elif isinstance(value, Iterable) and not isinstance(value, str):
            subdict2edict(value)
        setattr(args, key, value)
    return args


parser = argparse.ArgumentParser(description='DeepPointMap SLAM algorithm')

parser.add_argument('--name',                   default='DeepPointMap',         type=str,
                    help='Name of the model')
parser.add_argument('--version',                default='v1.0',                 type=str,
                    help='Version of the model')
parser.add_argument('--mode',                   default='train',                type=str,
                    choices=['train', 'infer'])
parser.add_argument('--checkpoint', '-ckpt',    default='',                     type=str,
                    help='Training checkpoint file')
parser.add_argument('--weight', '-w',           default='',                     type=str,
                    help='Model pre-training weight')
parser.add_argument('--yaml_file', '-yaml',     default='',                     type=str,
                    help='Yaml file which contains config parameters')
parser.add_argument('--num_workers',            default=4,                      type=int,
                    help='Number of threads used for parallel data loading')
parser.add_argument('--thread_safety',          default=False,                  action='store_true',
                    help='Whether the data loading method is thread safety')
parser.add_argument('--use_cuda',               default='True',                 type=str_to_bool,
                    help='Using cuda to accelerate calculations')
parser.add_argument('--gpu_index',              default='0',                    type=str,
                    help='Index of gpu')
parser.add_argument('--use_ddp',                default=False,                  action='store_true',
                    help='Use distributed data parallel during training')
parser.add_argument('--local_rank',             default=0,                      type=int,
                    help='Local device id on current node, only valid in DDP mode before torch1.9.0')
parser.add_argument('--word_size',              default=1,                      type=int,
                    help='Total number of GPUs used in DDP mode')
parser.add_argument('--infer_src',              default=[],                     type=list,
                    help='Sequential pcd data director list for inference')
parser.add_argument('--infer_tgt',              default='log_infer',                     type=str,
                    help='Inference output director')
parser.add_argument('--multi_thread', '-mt',    default=False,                  action='store_true',
                    help='Using multi-thread asynchronous pipeline to accelerating inference')
parser.add_argument('--use_ros', '-ros',    default=False,                  action='store_true',
                    help='Inference on ros or not (experimental)')


# Use yaml for these paras
parser.add_argument('--dataset',                help='Dataset used for training or inference')
parser.add_argument('--transforms',             help='Data transformation methods, including preprocessing and augment')
parser.add_argument('--encoder',                help='Parameters for DMP Encoder network structure')
parser.add_argument('--decoder',                help='Parameters for DMP Decoder network structure')
parser.add_argument('--train',                  help='Parameters for controlling the training method')
parser.add_argument('--loss',                   help='Parameters for calculating overall loss')
parser.add_argument('--slam_system',            help='Parameters for full slam system, including frontend and backend')
