import os
import sys
import yaml
from parameters import *

args = parser.parse_args()
assert args.use_ddp == False
if args.use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
sys.path.insert(1, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import colorlog as logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import warnings

warnings.filterwarnings("ignore")

import torch
torch.manual_seed(42)
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from system.core import SlamSystem
from dataloader.body import BasicAgent
from dataloader.transforms import PointCloudTransforms
from network.encoder.encoder import Encoder
from network.decoder.decoder import Decoder


def main():
    # Load yaml and prepare platform
    global args
    if not os.path.exists(args.yaml_file):
        raise FileNotFoundError(f'yaml_file is not found: {args.yaml_file}')
    logger.info(f'Loading config from \'{args.yaml_file}\'...')
    with open(args.yaml_file, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, yaml.FullLoader)
    args = update_args(args, cfg)
    if not args.thread_safety:
        torch.multiprocessing.set_start_method('spawn')
        logger.warning(f'The start method of torch.multiprocessing has been set to \'spawn\'')
    if args.use_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        gpus = list(range(torch.cuda.device_count()))
        torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    else:
        args.device = torch.device('cpu')

    # Init models and load weights
    logger.info('Preparing model...')
    encoder = Encoder(args=args)
    decoder = Decoder(args=args)
    if(os.path.exists(args.weight) == False):
        logger.error(f'weight file not exists: {args.weight}!')
        raise FileNotFoundError(f'weight file {args.weight} not found!')
    else:
        logger.info(f'Load weight from \'{args.weight}\'')
        weights = torch.load(args.weight, map_location='cpu')
        encoder.load_state_dict(weights['encoder'], strict=True)
        decoder.load_state_dict(weights['decoder'], strict=False)
        logger.info(f'Initialization completed, device = \'{args.device}\'')
        
        
        
    # Init data-transform
    logger.info('Preparing data...')
    transforms = PointCloudTransforms(args=args, mode='infer')

    # For each sequence...
    for i, data_root in enumerate(args.infer_src):
        if(isinstance(data_root,str)):
            data_root=[data_root]
        str_list = list(filter(lambda dir: os.path.exists(dir), data_root))
        if (len(str_list) == 0):
            logger.error(f"dir in source '{data_root}' ({i}) not found, SKIP!")
            continue
        else:
            logger.info(f"current sequence ({i}) loaded from '{data_root}'")
            data_root = str_list[0]
        dataset = BasicAgent(root=data_root, reader='auto')
        dataset.set_independent(data_transforms=transforms)
        
        
        # Create result dir and save yaml
        save_root = os.path.join(args.infer_tgt, f'Seq{i:02}')
        os.makedirs(save_root, exist_ok=True)
        with open(os.path.join(save_root, 'settings.yaml'), 'w+', encoding='utf-8') as arg_file:
            args_dict = sorted(args._get_kwargs())
            for k, v in args_dict:
                arg_file.write(f'{k}: {v}\n')

        # Prepare dataloader
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        infer_loop = tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, desc=f'{"dataloader":<12s}')

        # Init SLAM system
        slam_system = SlamSystem(args=args, dpm_encoder=encoder, dpm_decoder=decoder, system_id=0, logger_dir=save_root)
        if args.multi_thread:
            # Feed data! (Multi-Thread / MT)
            slam_system.MT_Init()
            for frame_id, data in enumerate(infer_loop):
                slam_system.MT_Step(data)
            slam_system.MT_Done()
            slam_system.MT_Wait()

        else:
            # Feed data! (Single Thread)
            for frame_id, data in enumerate(infer_loop):
                code = slam_system.step(data)
                infer_loop.set_description_str(f"infer: [{code}]" + ", ".join([f"{name}:{time[0]:.3f}s" for name, time in slam_system.result_logger.log_time(window=50).items()]))
        slam_system.result_logger.save_trajectory('trajectory')
        slam_system.result_logger.save_posegraph('trajectory')
        slam_system.result_logger.draw_trajectory('trajectory', draft=False)
        slam_system.result_logger.save_map('trajectory')
        logger.info(f'Sequence {i} End, Time = '+", ".join([f"{name}:{time[0]:.3f}/{time[1]:.3f}s" for name, time in slam_system.result_logger.log_time().items()]))


if __name__ == "__main__":
    main()
    logger.info('Done.')
