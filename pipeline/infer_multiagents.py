import os
import sys
from typing import List
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

# torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from copy import deepcopy

from system.core import CloudSystem, AgentSystem
from dataloader.body import BasicAgent
from dataloader.transforms import PointCloudTransforms
from network.encoder.encoder import Encoder
from network.decoder.decoder import Decoder
from system.modules.utils import Communicate_Module

AGENT_NUMBER = 3


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
    if (os.path.exists(args.weight) == False):
        logger.error(f'weight file not exists: {args.weight}!')
        raise FileNotFoundError(f'weight file {args.weight} not found!')
    else:
        logger.info(f'Load weight from \'{args.weight}\'')
        weights = torch.load(args.weight, map_location='cpu')
        encoder.load_state_dict(weights['encoder'])
        decoder.load_state_dict(weights['decoder'])
        logger.info(f'Initialization completed, device = \'{args.device}\'')

    # Init data-transform
    logger.info('Preparing data...')
    transforms = PointCloudTransforms(args=args, mode='infer')

    # For each sequence...
    for i, data_root in enumerate(args.infer_src):
        if (isinstance(data_root, str)):
            data_root = [data_root]
        str_list = list(filter(lambda dir: os.path.exists(dir), data_root))
        if (len(str_list) == 0):
            logger.error(f"sequence in source '{data_root}' ({i}) not found, SKIP!")
            continue
        else:
            logger.info(f"loading data '{data_root}' ({i})")
            data_root = str_list[0]

        # Create result dir and save yaml
        save_root = os.path.join(args.infer_tgt, f'Seq{i:02}')
        os.makedirs(save_root, exist_ok=True)
        with open(os.path.join(save_root, 'settings.yaml'), 'w+', encoding='utf-8') as arg_file:
            args_dict = sorted(args._get_kwargs())
            for k, v in args_dict:
                arg_file.write(f'{k}: {v}\n')

        communicate_module = Communicate_Module()

        cloud_system = CloudSystem(args=args, dpm_encoder=deepcopy(encoder), dpm_decoder=deepcopy(decoder), logger_dir=save_root, comm_module=communicate_module)
        cloud_system.start()

        agent_systems: List[AgentSystem] = []
        for agent_index in range(AGENT_NUMBER):
            dataset = BasicAgent(root=data_root, reader='auto', split_num=AGENT_NUMBER, split_index=agent_index)
            dataset.set_independent(data_transforms=transforms)
            # Prepare dataloader
            dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            # Init SLAM system
            agent_system = AgentSystem(
                args=args,
                dpm_encoder=deepcopy(encoder),
                dpm_decoder=deepcopy(decoder),
                system_id=agent_index + 1,
                logger_dir=save_root,
                comm_module=communicate_module,
            )
            # Feed data!
            agent_system.start(dataloader)
            agent_systems.append(agent_system)

        for agent_system in agent_systems:
            agent_system.wait()
            agent_system.result_logger.save_trajectory(f'trajectory_{agent_system.system_id}')
            agent_system.result_logger.save_posegraph(f'trajectory_{agent_system.system_id}')
            agent_system.result_logger.draw_trajectory(f'trajectory_{agent_system.system_id}', draft=False)
            agent_system.result_logger.save_map(f'trajectory_{agent_system.system_id}')
            agent_system.comm_module.send_message(caller=agent_system.comm_id, callee=0, command='AGENT_QUIT', message=None)
        communicate_module.send_message(caller=0, callee=0, command='QUIT', message=None)
        cloud_system.wait()


if __name__ == "__main__":
    main()
    logger.info('Done.')
