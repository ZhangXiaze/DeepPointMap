import colorlog as logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from glob import glob
from tqdm import tqdm
from typing import Dict, List, Tuple, Callable, Union
from dataloader.heads.auto import PointCloudReader
from dataloader.heads.npz import NPZReader
from dataloader.heads.npy import NPYReader
from dataloader.heads.bin import BinReader
from dataloader.heads.pcd import PcdReader

READER: Dict[str, PointCloudReader] = {
    'auto': PointCloudReader,
    'npz': NPZReader,
    'npy': NPYReader,
    'bin': BinReader,
    'pcd': PcdReader,
}


def get_length_range(l):
    length_range = [0]
    for i in l:
        length_range.append(len(i) + length_range[-1])
    return length_range


class SlamDatasets(Dataset):

    def __init__(self,
                 args,
                 data_transforms: Callable = nn.Identity()):

        Dataset.__init__(self)
        self.args = args
        self.dataset_cfg = self.args.dataset
        self.registration_cfg = self.args.train.registration
        self.loop_detection_cfg = self.args.train.loop_detection
        self.data_transforms = data_transforms

        self.dataset_list = self.load_dataset()  # [BasicDataset1, BasicDataset2, ...]

        self.pcd_range = get_length_range(self.dataset_list)
        self.pcd_range = torch.tensor(self.pcd_range, dtype=torch.int32)

        self.frame_distance = get_frame_dis(self.dataset_list)

        self._getitem_method = self._getitem_registration
        self.collate_fn = self.map_collate_fn

    def __getitem__(self, item):
        return self._getitem_method(item)

    def _getitem_loop_detection(self, item):
        dataset_id = torch.sum(self.pcd_range <= item) - 1 
        offset = item - self.pcd_range[dataset_id] 
        curren_dataset = self.dataset_list[dataset_id]

        # (pcd, lidar_seg), ego_rotation, ego_translation, images
        # Load first frame
        frame1 = curren_dataset[offset]

        scene_id, frame_offset = curren_dataset.get_frame_order(offset)
        frame_dis = self.frame_distance[dataset_id][scene_id][frame_offset]

        '''
        0    .50      .75     1
          <d     d~2d     >2d
        '''
        s = random.random()
        d = self.loop_detection_cfg.distance
        if s < 0.5:
            dis_mask = frame_dis <= d
        elif s < 0.75:
            dis_mask = (frame_dis > d) & (frame_dis <= 2 * d)
        else:
            dis_mask = frame_dis > 2 * d
        optional_pair_offset = torch.nonzero(dis_mask).squeeze(1) - frame_offset 
        optional_pair_offset = optional_pair_offset.tolist()
        if len(optional_pair_offset) > 0:
            pair_offset = random.choice(optional_pair_offset)
        else:
            pair_offset = 0
        frame2 = curren_dataset[offset + pair_offset]
        frame1 = self.data_transforms(frame1)
        frame2 = self.data_transforms(frame2)
        return *frame1, *frame2

    def _getitem_registration(self, index: int) -> Tuple[List, dict]:
        S = random.randint(2, self.registration_cfg.K) 
        if random.random() < 0.34:
            S = 2
        if self.registration_cfg.fill:
            num_map = self.registration_cfg.K_max // S
        else:
            num_map = 1
        info = dict(dsf_index=[], refined_SE3_file=[], num_map=num_map)

        frame_list = []
        for i in range(num_map):
            if i == 0:
                frame_list += self._map_query(index, K=S, info=info)
            else:
                rand_index = random.randint(0, self.__len__() - 1)
                frame_list += self._map_query(rand_index, K=S, info=info)

        return frame_list, info

    def _map_query(self, index: int, K: int, info: dict) -> List:

        dataset_id = (torch.sum(self.pcd_range <= index) - 1).item() 
        offset = index - self.pcd_range[dataset_id] 
        curren_dataset = self.dataset_list[dataset_id]
        scene_id, frame_offset = curren_dataset.get_frame_order(offset)
        frame_dis = self.frame_distance[dataset_id][scene_id][frame_offset]  

        dis_mask = frame_dis <= self.registration_cfg.distance - 0.25
        if dis_mask.sum() <= K:
            optional_frame_offsets = torch.nonzero(dis_mask).squeeze(1) - frame_offset  
            optional_frame_offsets = optional_frame_offsets.tolist()
            optional_frame_offsets.remove(0)  
            if len(optional_frame_offsets) == 0:
                optional_frame_offsets.append(0)
            optional_frame_offsets = optional_frame_offsets * (K // len(optional_frame_offsets) + 1)  
            map_frame_offsets = random.sample(optional_frame_offsets, k=K - 1)
            map_frame_offsets.insert(0, 0)
        else:
            optional_frame_offsets = torch.nonzero(dis_mask).squeeze(1) - frame_offset 
            optional_frame_offsets = optional_frame_offsets.tolist()
            optional_frame_offsets.remove(0)  
            map_frame_offsets = random.sample(optional_frame_offsets, k=K - 1)
            map_frame_offsets.insert(0, 0)
        info['dsf_index'] += [(dataset_id, scene_id, frame_offset + off) for off in map_frame_offsets]
        if 'carla' in curren_dataset.name.lower():
            refined_SE3_file = ''
        else:
            refined_SE3_file = os.path.join(curren_dataset.scene_list[scene_id].root, 'refined_SE3.pkl')
        info['refined_SE3_file'].append(refined_SE3_file)

        frame_list = []
        for map_frame_offset in map_frame_offsets:
            frame = curren_dataset[offset + map_frame_offset]
            frame = self.data_transforms(frame)
            frame_list.append(frame)
        return frame_list

    @staticmethod
    def map_collate_fn(batch):
        frame_list, info = batch[0]
        batch_data_list = []
        for data in zip(*frame_list):
            batch_data_list.append(torch.stack(data, dim=0))
        return *batch_data_list, info

    def __len__(self):
        return self.pcd_range[-1]

    def load_dataset(self):
        dataset_list: List[BasicDataset] = []
        for dataset_dict in self.dataset_cfg:
            name = dataset_dict.name
            root = dataset_dict.root
            scenes = dataset_dict.scenes
            reader_cfg = dataset_dict.reader
            reader = READER[reader_cfg.type](**reader_cfg.get('kwargs', {}))
            basic_dataset = BasicDataset(root=root, reader=reader, scenes=scenes, name=name.lower(), args=self.args)
            dataset_list.append(basic_dataset)
            logger.info(f'Load {name} successfully: \'{basic_dataset.root}\'')
        return dataset_list

    def get_seq_range(self):
        real_range = [0]
        for dataset in self.dataset_list:
            for scene in dataset.scene_list:
                for agent in scene.agent_list:
                    real_range.append(len(agent) + real_range[-1])
        return torch.tensor(real_range, dtype=torch.int32)

    def get_datasets(self):
        return self.dataset_list

    @property
    def seq_begin_list(self):
        return self.get_seq_range()

    def get_data_source(self, item):
        dataset_id = torch.sum(self.pcd_range <= item) - 1 
        return self.dataset_list[dataset_id]

    def registration(self):
        self._getitem_method = self._getitem_registration
        self.collate_fn = self.map_collate_fn

    def loop_detection(self):
        self._getitem_method = self._getitem_loop_detection
        self.collate_fn = None

    def __repr__(self):
        print('=' * 50)
        print(f'SlamDatasets: num_datasets={len(self.dataset_list)}\n'
              f'    |')
        for dataset in self.dataset_list:
            print(f'    |——{dataset.name}\n'
                  f'    |   |——train: num_scenes={len(dataset.scene_list)} | num_frames={dataset.pcd_range[-1]}\n'
                  f'    |')
        print('=' * 50)

    def __str__(self):
        my_str = ''
        my_str += ('=' * 50 + '\n')
        my_str += (f'SlamDatasets: num_datasets={len(self.dataset_list)}\n'
                   f'    |\n')
        for dataset in self.dataset_list:
            my_str += (f'    |——{dataset.name}\n'
                       f'    |   |——train: num_scenes={len(dataset.scene_list)} | num_frames={dataset.pcd_range[-1]}\n'
                       f'    |\n')
        my_str += ('=' * 50)
        return my_str


class BasicDataset:
    """
    dataset
        |--scenes
             |--00
             |--01
             |--02
                 |--agent 0
                 |--agent 1
                 |--agent 2
                        |--0.npz
                        |--1.npz
                        |--2.npz
    """

    def __init__(self,
                 args,
                 root: str,
                 reader: NPZReader,
                 scenes: list,
                 name: str,
                 ):
        self.args = args
        self.root = root
        self.scenes = scenes
        self.name = name

        if not isinstance(self.root, str) or not os.path.isdir(self.root):
            raise NotADirectoryError(f'\'{self.root}\' is not a directory')

        self.scene_list: List[BasicScene] = []
        for scene_name in self.scenes:
            scene_root = os.path.join(self.root, scene_name)
            if not os.path.isdir(scene_root):
                raise NotADirectoryError(f'\'{scene_root}\' is not a directory')
            self.scene_list.append(BasicScene(root=scene_root, reader=reader, parent=self, args=self.args))
        self.pcd_range = get_length_range(self.scene_list)
        self.pcd_range = torch.tensor(self.pcd_range, dtype=torch.int32)

    def __getitem__(self, item):
        scene_id = torch.sum(self.pcd_range <= item) - 1  
        offset = item - self.pcd_range[scene_id] 
        return self.scene_list[scene_id][offset]

    def __len__(self):
        return self.pcd_range[-1]

    def get_scenes(self):
        return self.scene_list

    def get_frame_order(self, item):
        scene_id = torch.sum(self.pcd_range <= item) - 1 
        offset = item - self.pcd_range[scene_id]  
        return scene_id.item(), offset.item()


class BasicScene:


    def __init__(self,
                 args,
                 root: str,
                 reader: NPZReader,
                 parent: BasicDataset,
                 ):

        self.root = root
        self.args = args
        self.parent = parent

        self.agent_list: List[BasicAgent] = []
        for agent_name in sorted(os.listdir(self.root)):
            agent_root = os.path.join(self.root, agent_name)
            if os.path.isdir(agent_root):
                self.agent_list.append(BasicAgent(root=agent_root, reader=reader, parent=self))

        self.pcd_range = get_length_range(self.agent_list)
        self.pcd_range = torch.tensor(self.pcd_range, dtype=torch.int32)

    def __getitem__(self, item):
        agent_id = torch.sum(self.pcd_range <= item) - 1  
        offset = item - self.pcd_range[agent_id]  
        return self.agent_list[agent_id][offset]

    def __len__(self):
        return self.pcd_range[-1]


class BasicAgent(Dataset):

    def __init__(self,
                 root: str,
                 reader: Union[PointCloudReader, str],
                 parent: BasicScene = None,
                 split_num: int = 1,
                 split_index: int = 0
                 ):

        Dataset.__init__(self)
        self.root = root  # './../../Dataset/KITTI_Odometry_MultiAgent\\0\\0\*.npz'
        self.reader = reader
        self.parent = parent
        self.data_transforms = None
        file_name_list = glob(os.path.join(self.root, '*.*'))
        file_type = set([os.path.splitext(i)[1] for i in file_name_list])
        assert len(file_type) <= 1, 'The root can only contain files of the SAME type'
        file_type = file_type.pop()[1:]
        if self.reader == 'auto':
            self.reader = READER[file_type]()
        file_name_list = sorted(file_name_list, key=lambda s: int(os.path.basename(s).split('.')[0]))

        if split_num > 1:
            total_len = len(file_name_list)
            agent_ratio = 1 / split_num
            overlap_ratio = 1 / 20  # 5% overlapped frames
            start_ratio = max(agent_ratio * split_index - overlap_ratio, 0.0)
            end_ratio = min(agent_ratio * (split_index + 1) + overlap_ratio, 1.0)
            self.file_list = file_name_list[int(total_len * start_ratio):int(total_len * end_ratio)]
        else:
            self.file_list = file_name_list

    def __getitem__(self, item):
        data = self.reader(self.file_list[item])  # (pcd, seg), R, T, imgs
        if self.data_transforms is not None:
            data = self.data_transforms(data)
        return data

    def __len__(self):
        return len(self.file_list)

    def set_independent(self, data_transforms: Callable):
        self.data_transforms = data_transforms


def get_frame_dis(dataset_list: List[BasicDataset]) -> List[List[torch.Tensor]]:
    frame_distance = []
    for i, dataset in enumerate(dataset_list):
        dataset_frame_dis = []
        for j, scene in enumerate(dataset.scene_list):
            frame_files = []
            for agent in scene.agent_list:
                frame_files += agent.file_list

            frame_dis_file = os.path.join(scene.root, 'frame_dis.npy')
            cache_right = False
            if os.path.exists(frame_dis_file):
                frame_dis: np.ndarray = np.load(frame_dis_file).astype(np.float32)
                if frame_dis.shape[0] == frame_dis.shape[1] == len(frame_files):
                    cache_right = True
            if not cache_right:
                frame_poses = []
                loop = tqdm(frame_files, total=len(frame_files), leave=False, dynamic_ncols=True)
                loop.set_description(f'Building \'frame_dis.npy\' | Dataset No.{i + 1} | Scene No.{j + 1}')
                for frame_file in loop:
                    with np.load(frame_file, allow_pickle=True) as npz:
                        frame_pose = npz['ego_translation'].squeeze(1).astype(np.float32)  # (3, 1), f32
                        frame_poses.append(frame_pose)
                frame_poses = np.stack(frame_poses, axis=0)  # (N, 3)
                frame_dis = np.linalg.norm(
                    x=(np.expand_dims(frame_poses, axis=1) - np.expand_dims(frame_poses, axis=0)), ord=2, axis=-1)

                np.save(file=frame_dis_file, arr=frame_dis)
                logger.info(f'File \'frame_dis\' has been saved in {frame_dis_file}')

            frame_dis = torch.from_numpy(frame_dis).half()
            dataset_frame_dis.append(frame_dis)
        frame_distance.append(dataset_frame_dis)
    return frame_distance

