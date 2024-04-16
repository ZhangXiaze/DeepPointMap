import colorlog as logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
import time
import zipfile
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast as autocast
from glob import glob
from tqdm import tqdm
from collections import OrderedDict
from pipeline.modules.utils import Recorder, fakecast, Optimizer, Scheduler, try_load_state_dict
from pipeline.modules.model_pipeline import DeepPointModelPipeline
from utils.device import move_to_device
from dataloader.body import SlamDatasets

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.manual_seed(42)


class Trainer:

    def __init__(self, args, dataset: SlamDatasets, model: DeepPointModelPipeline):
        self.args = args
        self.train_cfg = args.train
        self.dataset = dataset
        self.model = model
        self.stage_epoch = (self.train_cfg.registration.num_epochs, self.train_cfg.loop_detection.num_epochs)

        self.optimizer = None
        self.scheduler = None
        self.dataloader = None
        self.sampler = None
        self.writer = None
        self.log_interval = None
        self.epoch = 1
        self.step = 1
        if self.train_cfg.auto_cast:
            self.cast = autocast
        else:
            self.cast = fakecast
        self.log = f'{self.args.name}{self.args.version}_config={os.path.split(self.args.yaml_file)[1]}'
        self.save_root = os.path.join('log_train', self.log)
        self.is_main_process = not (self.args.use_ddp and self.args.local_rank != 0)

        if args.checkpoint != '':
            self.load_checkpoint(args.checkpoint)
        elif args.weight != '':
            self.load_weight(args.weight)
        else:
            self.init_scratch()

        if self.is_main_process:
            os.makedirs(self.save_root, exist_ok=True)
            logger.info(f'save root = \'{self.save_root}\'')
            args_dict = sorted(args._get_kwargs())
            with open(os.path.join(self.save_root, 'settings.yaml'), 'w+', encoding='utf-8') as arg_file:
                for k, v in args_dict:
                    arg_file.write(f'{k}: {v}\n')
            code_files = [f for f in sorted(glob('./**/*.py', recursive=True)) if not os.path.basename(f).startswith('__')]
            zfile = zipfile.ZipFile(os.path.join(self.save_root, 'codes.zip'), mode='w', compression=zipfile.ZIP_DEFLATED, compresslevel=9)
            for f in code_files:
                zfile.write(f)
            zfile.close()
        s = f'Initialization completed, device = \'{self.args.device}\''
        if self.is_main_process:
            s += ' [MAIN PROCESS]'
        logger.info(s)
        if self.args.use_ddp:
            dist.barrier()

    def run(self):
        if self.epoch <= self.stage_epoch[0]:
            self.dataset.registration()
            batch_size = self.train_cfg.registration.batch_size
        else:
            self.dataset.loop_detection()
            batch_size = self.train_cfg.loop_detection.batch_size

        if self.args.use_ddp:
            self.sampler = DistributedSampler(self.dataset)
            self.dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size,
                                         num_workers=self.args.num_workers, sampler=self.sampler,
                                         collate_fn=self.dataset.collate_fn, pin_memory=True, drop_last=True)
        else:
            self.dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size,
                                         num_workers=self.args.num_workers, shuffle=True,
                                         collate_fn=self.dataset.collate_fn, pin_memory=True, drop_last=True)

        if self.is_main_process:
            self.writer = SummaryWriter(os.path.join('log_tb', self.log))
            train_record = Recorder()

        start_epoch = self.epoch
        for ep in range(start_epoch, sum(self.stage_epoch) + 1):
            self._epoch_begin(ep)

            train_metric = self.train_one_epoch()

            self.scheduler.step()

            if self.is_main_process:
                train_record.add_dict(train_metric)

                if ep % self.train_cfg.save_cycle == 0:
                    self.save()

            self.epoch += 1

        if self.is_main_process:
            self.save(finish=True)
            print(train_record.tostring())
        if self.args.use_ddp:
            dist.barrier()

    def _epoch_begin(self, ep):
        if self.args.use_ddp:
            dist.barrier()

        if ep == self.stage_epoch[0] + 1:
            self._next_stage() 

        if ep <= self.stage_epoch[0]:
            registration_cfg = self.train_cfg.registration
            if 'K_0' in registration_cfg.keys():
                K_0 = registration_cfg['K_0']
                K_mult = registration_cfg['K_mult']
                mult_epoch = registration_cfg['mult_epoch']
                times = 0
                for i in mult_epoch:
                    if ep >= i:
                        times += 1
                registration_cfg['K'] = K_0 * (K_mult ** times)
            batch_size = registration_cfg.batch_size
            if self.is_main_process:
                self.writer.add_scalar("runtime/K", registration_cfg['K'], ep)
        else:
            batch_size = self.train_cfg.loop_detection.batch_size

        if self.is_main_process:
            self.writer.add_scalar("runtime/learning_rate", self.optimizer.param_groups[0]['lr'], ep)

        if self.args.use_ddp:
            self.sampler.set_epoch(ep)
            log_interval = (self.train_cfg.log_cycle / self.args.word_size) // batch_size
        else:
            log_interval = self.train_cfg.log_cycle // batch_size
        self.log_interval = int(max(log_interval, 1))

    def train_one_epoch(self):
        start_time = time.time()
        self.model.train()
        step_count = 0
        log_interval = self.log_interval
        epoch_metrics = dict()

        if self.args.use_ddp:
            dist.barrier()

        loop = tqdm(self.dataloader, total=len(self.dataloader), leave=False, dynamic_ncols=True)
        loop.set_description('train')
        for data in loop:
            step_count += 1
            data = move_to_device(data, device=self.args.device, non_blocking=True)

            with self.cast():
                loss, metric = self.model(*data)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loop.set_postfix_str(' | '.join(f'{k}={v:2.4f}' for k, v in metric.items()))

            if self.is_main_process:
                for metric_name, metric_value in metric.items():
                    epoch_metrics.setdefault(metric_name, []).append(metric_value)

                if step_count % log_interval == 0:
                    for label, metric_list in epoch_metrics.items():
                        self.writer.add_scalar(f"train/step_{label}", sum(metric_list[-log_interval:]) / log_interval,
                                               self.step)
            self.step += 1

        # Epoch ends
        if not self.is_main_process:
            return None

        summary_str = ''
        summary_metric = {}
        for label, metric_list in epoch_metrics.items():
            self.writer.add_scalar(f"train/epoch_{label}", sum(metric_list) / len(metric_list), self.epoch)
            summary_str += f'{label} = {sum(metric_list) / len(metric_list):6.4f} | '
            summary_metric[label] = sum(metric_list) / len(metric_list)

        cost_time = time.time() - start_time
        cost_m, cost_s = divmod(cost_time, 60)
        cost_h, cost_m = divmod(cost_m, 60)
        logger.info(f'Train Epoch {self.epoch:>4d} | ' + summary_str +
                    f'Time = {int(cost_h)}h:{int(cost_m):02d}m:{cost_s:04.1f}s')
        return summary_metric

    def save(self, finish=False):
        if self.args.use_ddp:
            encoder_state_dict = self.model.module.encoder.state_dict()
            decoder_state_dict = self.model.module.decoder.state_dict()
        else:
            encoder_state_dict = self.model.encoder.state_dict()
            decoder_state_dict = self.model.decoder.state_dict()
        if not finish:
            state = {
                'encoder': encoder_state_dict,
                'decoder': decoder_state_dict,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.epoch,
                'step': self.step,
            }
            file_path = os.path.join(self.save_root, f'{self.args.name}{self.args.version}_epoch{self.epoch}.ckpt')
        else:
            state = {
                'encoder': encoder_state_dict,
                'decoder': decoder_state_dict,
            }
            file_path = os.path.join(self.save_root, f'{self.args.name}{self.args.version}.pth')
        torch.save(state, file_path)

    def init_scratch(self):
        optimizer = Optimizer(self.train_cfg.registration.optimizer)
        scheduler = Scheduler(self.train_cfg.registration.scheduler)
        self.model.registration()
        if self.args.use_ddp:
            self.model = DistributedDataParallel(self.model.cuda(self.args.local_rank),
                                                 device_ids=[self.args.local_rank],
                                                 output_device=self.args.local_rank)
        else:
            self.model = self.model.to(self.args.device)
        self.optimizer = optimizer(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.scheduler = scheduler(self.optimizer)
        if self.is_main_process:
            logger.info(f'Training from scratch')

    def load_checkpoint(self, checkpoint: str):
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f'checkpoint file \'{checkpoint}\' is not found.')
        checkpoint_file_path = checkpoint

        if self.args.use_ddp:
            checkpoint = torch.load(checkpoint, map_location=f'cuda:{self.args.local_rank}')
        else:
            checkpoint = torch.load(checkpoint, map_location=self.args.device)

        # Load model
        self.epoch = checkpoint['epoch'] + 1
        if self.is_main_process:
            logger.info(f"Load epoch, current = {self.epoch}")
        self.step = checkpoint['step']
        if self.is_main_process:
            logger.info(f"Load step, current = {self.step}")
        encoder_state_dict = checkpoint['encoder']
        try_load_state_dict(self.model.encoder, encoder_state_dict, 'encoder', log=self.is_main_process)
        decoder_state_dict = checkpoint['decoder']
        try_load_state_dict(self.model.decoder, decoder_state_dict, 'decoder', log=self.is_main_process)

        if self.epoch <= self.stage_epoch[0]:
            self.model.registration()
            optimizer = Optimizer(self.train_cfg.registration.optimizer)
            scheduler = Scheduler(self.train_cfg.registration.scheduler)
        else:
            self.model.loop_detection()
            optimizer = Optimizer(self.train_cfg.loop_detection.optimizer)
            scheduler = Scheduler(self.train_cfg.loop_detection.scheduler)
        if self.args.use_ddp:
            self.model = DistributedDataParallel(self.model.cuda(self.args.local_rank),
                                                 device_ids=[self.args.local_rank],
                                                 output_device=self.args.local_rank)
        else:
            self.model = self.model.to(self.args.device)
        self.optimizer = optimizer(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.scheduler = scheduler(self.optimizer)

        if self.epoch != self.stage_epoch[0] + 1:
            try_load_state_dict(self.optimizer, checkpoint['optimizer'], 'optimizer', log=self.is_main_process)
            try_load_state_dict(self.scheduler, checkpoint['scheduler'], 'scheduler', log=self.is_main_process)
        if self.is_main_process:
            logger.info(f'Load checkpoint done. \'{checkpoint_file_path}\'')

    def load_weight(self, weight: str):
        if not os.path.exists(weight):
            raise FileNotFoundError(f'weight file \'{weight}\' is not found.')
        weight_file_path = weight

        if self.args.use_ddp:
            weight = torch.load(weight, map_location=f'cuda:{self.args.local_rank}')
        else:
            weight = torch.load(weight, map_location=self.args.device)

        encoder_state_dict = weight['encoder']
        try_load_state_dict(self.model.encoder, encoder_state_dict, 'encoder', log=self.is_main_process)
        decoder_state_dict = weight['decoder']
        try_load_state_dict(self.model.decoder, decoder_state_dict, 'decoder', log=self.is_main_process)
        self.init_scratch()
        if self.is_main_process:
            logger.info(f'Load specific weight from \'{weight_file_path}\'')

    def _next_stage(self):
        self.dataset.loop_detection()
        batch_size = self.train_cfg.loop_detection.batch_size
        if self.args.use_ddp:
            model = self.model.module
            model.loop_detection()
            self.model = DistributedDataParallel(model.cuda(self.args.local_rank),
                                                 device_ids=[self.args.local_rank],
                                                 output_device=self.args.local_rank)
            self.sampler = DistributedSampler(self.dataset)
            self.dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size,
                                         num_workers=self.args.num_workers, sampler=self.sampler,
                                         collate_fn=self.dataset.collate_fn, pin_memory=True, drop_last=True)
        else:
            self.model.loop_detection()
            self.dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size,
                                         num_workers=self.args.num_workers, shuffle=True,
                                         collate_fn=self.dataset.collate_fn, pin_memory=True, drop_last=True)
        optimizer = Optimizer(self.train_cfg.loop_detection.optimizer)
        scheduler = Scheduler(self.train_cfg.loop_detection.scheduler)
        self.optimizer = optimizer(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.scheduler = scheduler(self.optimizer)
        if self.is_main_process:
            logger.info(f'Convert the training stage from registration to loop-detection')

    @staticmethod
    def add_module(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not k.startswith('module.'):
                k = 'module.' + k
            new_state_dict[k] = v
        return new_state_dict

    @staticmethod
    def remove_module(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        return new_state_dict
