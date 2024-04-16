import os
import sys
import yaml
from parameters import *
args = parser.parse_args()
if not args.use_ddp and args.use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
sys.path.insert(1, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import colorlog as logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from dataloader.body import SlamDatasets
from dataloader.transforms import PointCloudTransforms
from network.encoder.encoder import Encoder
from network.decoder.decoder import Decoder
from network.loss import RegistrationLoss
from modules.model_pipeline import DeepPointModelPipeline
from modules.trainer import Trainer


def main():
    global args
    if not os.path.exists(args.yaml_file):
        raise FileNotFoundError(f'yaml_file \'{args.yaml_file}\' is not found!')
    logger.info(f'Loading config from \'{args.yaml_file}\'...')
    with open(args.yaml_file, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, yaml.FullLoader)
    args = update_args(args, cfg)
    if not args.thread_safety:
        torch.multiprocessing.set_start_method('spawn')
        logger.warning(f'The start method of torch.multiprocessing has been set to \'spawn\'')
    if args.use_ddp and torch.cuda.is_available():
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', rank=args.local_rank, world_size=args.word_size)
        torch.cuda.set_device(args.device)
    elif args.use_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        gpus = list(range(torch.cuda.device_count()))
        torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    else:
        args.device = torch.device('cpu')

    logger.info('Preparing data...')
    transforms = PointCloudTransforms(args=args, mode='train')
    if args.use_ddp:
        if args.local_rank == 0:
            dataset = SlamDatasets(args=args, data_transforms=transforms)
            print(dataset)
        dist.barrier()
        if args.local_rank != 0:
            dataset = SlamDatasets(args=args, data_transforms=transforms)
    else:
        dataset = SlamDatasets(args=args, data_transforms=transforms)
        print(dataset)

    logger.info('Preparing model...')
    encoder = Encoder(args=args)
    decoder = Decoder(args=args)
    criterion = RegistrationLoss(args=args)
    model = DeepPointModelPipeline(args=args, encoder=encoder, decoder=decoder, criterion=criterion)

    logger.info('Launching trainer...')
    trainer = Trainer(args=args, dataset=dataset, model=model)
    trainer.run()


if __name__ == "__main__":
    main()
    logger.info('Done.')
