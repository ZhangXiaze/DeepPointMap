import random
import math
import torch
from torch import Tensor as Tensor
import numpy as np
import open3d as o3d
import scipy.linalg as linalg
from scipy.spatial.transform import Rotation
from typing import List, Sequence, Literal, Union
try:
    from pytorch3d.ops import knn_points, sample_farthest_points, ball_query, knn_gather
    has_t3d = True
except:
    has_t3d = False


class PointCloud:

    def __init__(self,
                 xyz: Union[Tensor, np.ndarray],
                 rotation: Union[Tensor, np.ndarray] = None, translation: Union[Tensor, np.ndarray] = None,
                 norm: Union[Tensor, np.ndarray] = None, label: Union[Tensor, np.ndarray] = None,
                 image: Union[Tensor, np.ndarray] = None, uvd: Union[Tensor, np.ndarray] = None,
                 ) -> None:

        input_args = [xyz, rotation, translation, norm, label, image, uvd]
        for i in range(len(input_args)):
            arg = input_args[i]
            if arg is not None:
                arg = torch.from_numpy(arg)
                if arg.dtype == torch.float64:
                    arg = arg.float() 
                input_args[i] = arg
        xyz, rotation, translation, norm, label, image, uvd = input_args

        self.xyz = xyz
        self.nbr_point = xyz.shape[0]
        self.device = self.xyz.device

        self.R = rotation if rotation is not None else torch.eye(3, dtype=torch.float32, device=self.device)
        self.T = translation if translation is not None else torch.zeros(size=(3, 1), dtype=torch.float32, device=self.device)

        self.calib = torch.eye(4, dtype=torch.float32, device=self.device)

        self.norm = norm
        if norm is not None:
            self.has_norm = True
        else:
            self.has_norm = False

        self.label = label
        if label is not None:
            self.has_label = True
        else:
            self.has_label = False

        self.image = image
        if image is not None:
            self.has_image = True
        else:
            self.has_image = False

        self.uvd = uvd
        if uvd is not None:
            self.has_uvd = True
        else:
            self.has_uvd = False

    def to_tensor(self, use_norm: bool = False, use_uvd: bool = False, use_image: bool = False, use_calib: bool = False,
                  padding_to: int = -1):

        constitution = [self.xyz]
        if use_norm and self.has_norm:
            constitution.append(self.norm)
        if use_uvd and self.has_uvd:
            constitution.append(self.uvd)
        pcd = torch.concat(constitution, dim=1)
        if padding_to > 0:
            if self.nbr_point > padding_to:
                raise RuntimeError(
                    f'The number of Point Cloud ({self.nbr_point}) is greater than `padding_to` ({padding_to})')
            padding = torch.zeros(size=(padding_to - self.nbr_point, pcd.shape[1]), device=self.device)
            pcd = torch.cat((pcd, padding), dim=0)
            padding_mask = torch.zeros(size=(padding_to,), dtype=torch.bool, device=self.device)
            padding_mask[self.nbr_point:] = True
        else:
            padding_mask = torch.zeros(self.nbr_point, dtype=torch.bool, device=self.device)

        if use_image:
            if self.has_image:
                image = self.image
            else:
                image = torch.zeros(size=(1,), device=self.device)
            return pcd.T, self.R, self.T, image, padding_mask
        elif use_calib:
            return pcd.T, self.R, self.T, padding_mask, self.calib
        else:
            return pcd.T, self.R, self.T, padding_mask

    def apply_index(self, mask):
        scalable_args = [self.xyz, self.norm, self.label, self.uvd]
        for i in range(len(scalable_args)):
            arg = scalable_args[i]
            if arg is not None:
                scalable_args[i] = arg[mask]
        self.xyz, self.norm, self.label, self.uvd = scalable_args
        self.nbr_point = self.xyz.shape[0]

    def to_gpu(self):
        self.xyz, self.R, self.T, self.calib = self.xyz.cuda(), self.R.cuda(), self.T.cuda(), self.calib.cuda()
        if self.has_norm:
            self.norm = self.norm.cuda()
        if self.has_label:
            self.label = self.label.cuda()
        if self.has_image:
            self.image = self.image.cuda()
        if self.has_uvd:
            self.uvd = self.uvd.cuda()
        self.device = self.xyz.device

    def to_cpu(self):
        self.xyz, self.R, self.T, self.calib = self.xyz.cpu(), self.R.cpu(), self.T.cpu(), self.calib.cpu()
        if self.has_norm:
            self.norm = self.norm.cuda()
        if self.has_label:
            self.label = self.label.cpu()
        if self.has_image:
            self.image = self.image.cpu()
        if self.has_uvd:
            self.uvd = self.uvd.cpu()
        self.device = self.xyz.device


class Compose:

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, pcd: PointCloud) -> PointCloud:
        for t in self.transforms:
            pcd = t(pcd)
        return pcd

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class RandomChoice:

    def __init__(self, transforms, p=None):
        if p is not None and not isinstance(p, Sequence):
            raise TypeError("Argument p should be a sequence")
        self.transforms = transforms
        self.p = p

    def __call__(self, pcd: PointCloud) -> PointCloud:
        t = random.choices(self.transforms, weights=self.p)[0]
        return t(pcd)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += f"\n)(p={self.p})"
        return format_string


class GroundFilter:

    def __init__(self, img_len: int, img_width: int, grid_width: float, ground_height: float,
                 preserve_sparse_ground: bool = True):

        self.img_len = img_len
        self.img_width = img_width
        self.grid_width = grid_width
        self.ground_height = ground_height
        self.preserve_sparse_ground = preserve_sparse_ground

    def __call__(self, pcd: PointCloud) -> PointCloud:
        if self.ground_height <= 0:
            return pcd

        pointCloudsIn = pcd.xyz.cpu().clone().numpy()

        row_id = (pointCloudsIn[:, 0] / self.grid_width + self.img_len / 2).astype(np.int32)  # (N,)
        col_id = (pointCloudsIn[:, 1] / self.grid_width + self.img_width / 2).astype(np.int32)  # (N,)
        grid_id = row_id * self.img_width + col_id  
        dis_mask = (row_id >= 0) & (row_id < self.img_len) & (col_id >= 0) & (col_id < self.img_width)
        pointCloudsIn = pointCloudsIn[dis_mask]  # (M, 3)
        grid_id = grid_id[dis_mask]  # (M,)
        remained_ids = np.nonzero(dis_mask)[0]  # (M,)

        order = np.argsort(grid_id)
        pointCloudsIn = pointCloudsIn[order]
        grid_id = grid_id[order]
        remained_ids = remained_ids[order]

        all_grid_id, all_grid_cnt = np.unique(grid_id, return_counts=True) 
        grid_slices = np.cumsum(all_grid_cnt, axis=-1) 
        non_ground_ids = [] 
        sparse_ground_ids = [] 

        end = 0
        for grid_slice in grid_slices:
            begin = end
            end = grid_slice
            if end - begin < 3:
                continue
            grid_pcd = pointCloudsIn[begin: end, :]
            grid_ids = remained_ids[begin: end]
            height_diff = grid_pcd[:, 2].max() - grid_pcd[:, 2].min()

            if height_diff > self.ground_height:
                non_ground_ids.append(grid_ids)
            elif self.preserve_sparse_ground:
                sparse_ground_ids.append(grid_ids[:1])

        remained_ids = np.concatenate(non_ground_ids + sparse_ground_ids, axis=0)
        remained_ids = torch.from_numpy(remained_ids).to(pcd.device)
        pcd.apply_index(mask=remained_ids)
        return pcd


class OutlierFilter:

    def __init__(self, nb_neighbors: int, std_ratio: float):
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

    def __call__(self, pcd: PointCloud) -> PointCloud:
        if str(pcd.device).startswith('cuda') and has_t3d:
            pcd_xyz = pcd.xyz.unsqueeze(0)
            knn = knn_points(p1=pcd_xyz, p2=pcd_xyz, K=self.nb_neighbors + 1, return_sorted=True, return_nn=False)
            dists = torch.sqrt(knn.dists.squeeze(0)[:, 1:])
            points_dist = dists.mean(1)
            mean = points_dist.mean()
            std = points_dist.std()
            outlier_dist = mean + self.std_ratio * std
            mask = points_dist <= outlier_dist
            pcd.apply_index(mask)
        else:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.open3d.utility.Vector3dVector(pcd.xyz.numpy())
            mask = pcd_o3d.remove_statistical_outlier(nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio)[1]
            pcd.apply_index(mask)

        return pcd


class LowPassFilter:

    def __init__(self, normals_radius: float, normals_num: int, filter_std: float, flux: int = 2,
                 max_remain: int = -1):

        assert has_t3d
        self.normals_radius = normals_radius
        self.normals_num = normals_num
        self.filter_std = filter_std
        self.flux = flux
        self.max_remain = max_remain

    def __call__(self, pcd: PointCloud) -> PointCloud:
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.open3d.utility.Vector3dVector(pcd.xyz.cpu().numpy())
        pcd_o3d.estimate_normals(search_param=(o3d.geometry.KDTreeSearchParamRadius(radius=self.normals_radius)))
        normals = torch.tensor(np.asarray(pcd_o3d.normals), dtype=torch.float, device=pcd.device)  # (N, 3)

        xyz = pcd.xyz
        K = self.normals_num
        result = knn_points(p1=xyz[None], p2=xyz[None], K=K + 1)
        grouped_indices = result.idx[..., 1:]
        grouped_normals = knn_gather(x=normals[None], idx=grouped_indices)[0]  # (N, K, 3)

        normals_similarity = (grouped_normals @ normals.unsqueeze(-1)).squeeze(-1).abs()
        sim, _ = torch.topk(normals_similarity, k=self.flux, dim=-1)
        sim = sim.sum(1)
        mask = sim > (sim.mean() - self.filter_std * sim.std())
        if 0 < self.max_remain < mask.sum():
            _, mask = torch.topk(sim, k=self.max_remain)

        pcd.apply_index(mask)

        return pcd

    def __repr__(self):
        s = f'{self.__class__.__name__}(normals_radius={self.normals_radius}, normals_num={self.normals_num}, ' \
            f'filter_std={self.filter_std}, flux={self.flux}, max_remain={self.max_remain})'
        return s

    def __str__(self):
        return self.__repr__()


class VerticalCorrect:

    def __init__(self, angle: float):
        self.angle = angle

    def __call__(self, pcd: PointCloud) -> PointCloud:
        if self.angle == 0:
            return pcd

        xyz = pcd.xyz.cpu().numpy()

        rotation_axis = np.cross(xyz, [0, 0, 1])

        rotation_axis = rotation_axis / linalg.norm(rotation_axis, axis=1, keepdims=True)  # Normalize axis
        r = Rotation.from_rotvec(rotation_axis * self.angle, degrees=True)
        rotation_matrix = r.as_matrix().astype(np.float32)

        corrected_xyz = (rotation_matrix @ xyz[:, :, np.newaxis]).squeeze(-1)
        pcd.xyz = torch.from_numpy(corrected_xyz)
        return pcd


class VoxelSample:

    def __init__(self, voxel_size: float, retention: Literal['first', 'center'] = 'center', num: int = None):
        assert retention in ['first', 'center'], f'\'{retention}\' is not a supported retention method, ' \
                                                 f'please use \'first\' or \'center\''
        self.voxel_size = voxel_size
        self.retention = retention
        self.num = num

    def __call__(self, pcd: PointCloud) -> PointCloud:
        device = pcd.device
        pcd_xyz = pcd.xyz.cpu().numpy()

        xyz_min = np.min(pcd_xyz, axis=0)
        xyz_max = np.max(pcd_xyz, axis=0)
        X, Y, Z = ((xyz_max - xyz_min) / self.voxel_size).astype(np.int32) + 1

        relative_xyz = pcd_xyz - xyz_min
        voxel_xyz = (relative_xyz / self.voxel_size).astype(np.int32)
        voxel_id = (voxel_xyz[:, 0] + voxel_xyz[:, 1] * X + voxel_xyz[:, 2] * X * Y).astype(np.int32)

        if self.retention == 'center':
            dis = np.sum((relative_xyz - voxel_xyz * self.voxel_size - self.voxel_size / 2) ** 2, axis=-1)
            sorted_id = np.argsort(dis)
            voxel_id = voxel_id[sorted_id]
            pcd.apply_index(torch.from_numpy(sorted_id).to(device))

        _, unique_id, cnt = np.unique(voxel_id, return_index=True, return_counts=True)

        if self.num is not None and unique_id.shape[0] > self.num:
            cnt_topk_id = np.argpartition(cnt, kth=-self.num)[-self.num:]
            unique_id = unique_id[cnt_topk_id]

        pcd.apply_index(torch.from_numpy(unique_id).to(device))
        return pcd


class FarthestPointSample:

    def __init__(self, num):
        if not has_t3d:
            raise NotImplementedError('Module pytorch3d not found! '
                                      '\'FarthestPointSample\' is only supported by PyTorch3D')
        self.num = num

    def __call__(self, pcd: PointCloud) -> PointCloud:
        if pcd.nbr_point > self.num:
            points_xyz = pcd.xyz
            idx = sample_farthest_points(points=points_xyz.unsqueeze(0), K=self.num)[1][0]
            pcd.apply_index(idx)
        return pcd


class RandomSample:

    def __init__(self, num):
        self.num = num

    def __call__(self, pcd: PointCloud) -> PointCloud:
        if pcd.nbr_point > self.num:
            downsample_ids = torch.randperm(pcd.nbr_point, device=pcd.device)[:self.num]
            pcd.apply_index(downsample_ids)
        return pcd


class DistanceSample:

    def __init__(self, min_dis: float, max_dis: float):
        self.min_dis = min_dis
        self.max_dis = max_dis

    def __call__(self, pcd: PointCloud) -> PointCloud:
        dis = torch.norm(pcd.xyz, p=2, dim=1)
        mask = (self.min_dis <= dis) & (dis <= self.max_dis)
        pcd.apply_index(mask)
        return pcd


class CoordinatesNormalization:

    def __init__(self, ratio: float):
        self.ratio = ratio

    def __call__(self, pcd: PointCloud) -> PointCloud:
        pcd.xyz /= self.ratio
        return pcd


class RandomShuffle:

    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(self, pcd: PointCloud) -> PointCloud:
        if random.random() > self.p:
            return pcd
        shuffle_ids = torch.randperm(pcd.nbr_point, device=pcd.device)
        pcd.apply_index(shuffle_ids)
        return pcd


class RandomDrop:

    def __init__(self, max_ratio: float, p: float = 1.0):
        self.max_ratio = max_ratio
        self.p = p

    def __call__(self, pcd: PointCloud) -> PointCloud:
        if random.random() > self.p:
            return pcd
        drop_ratio = random.uniform(0, self.max_ratio)
        remained_ids = torch.rand(size=(pcd.nbr_point,), device=pcd.device) >= drop_ratio
        pcd.apply_index(remained_ids)
        return pcd


class RandomOcclusion:

    def __init__(self, angle_range: list, dis_range: list, max_num: int, p: float = 0.1):
        super().__init__()
        self.angle_range = angle_range  
        self.dis_range = dis_range 
        self.max_num = max_num 
        self.p = p

    def __call__(self, pcd: PointCloud) -> PointCloud:
        if random.random() > self.p:
            return pcd
        xyz = pcd.xyz
        device = pcd.device

        azimuth_angle = (torch.atan2(other=xyz[:, 0], input=xyz[:, 1])) * 180 / torch.pi
        distance = torch.norm(xyz, p=2, dim=1)
        mask = torch.ones(size=(pcd.nbr_point, ), dtype=torch.bool, device=device)

        num = random.randint(1, self.max_num)
        for i in range(num):

            angle, dis, direction = torch.rand(size=(3, ), device=device)
            angle = (angle * (self.angle_range[1] - self.angle_range[0]) + self.angle_range[0]) / (i + 1)  
            dis_threshold = dis * (self.dis_range[1] - self.dis_range[0]) + self.dis_range[0]  
            direction = direction * 360 - 180 

            angle_start, angle_end = direction, direction + angle
            if angle_end <= 180:
                shield_angle = (azimuth_angle >= angle_start) & (azimuth_angle <= angle_end)
            else:
                shield_angle = (azimuth_angle >= angle_start) | (azimuth_angle <= angle_end - 360)
            shield_dis = (distance >= dis_threshold)
            mask &= ~(shield_angle & shield_dis)

        pcd.apply_index(mask)
        return pcd


class RandomRT:

    def __init__(self, r_mean: float = 0, r_std: float = 3.14, t_mean: float = 0, t_std: float = 1,
                 p: float = 1.0, pair: bool = True):
        self.r_mean = r_mean
        self.r_std = r_std
        self.t_mean = t_mean
        self.t_std = t_std
        self.p = p
        self.pair = pair
        self.flag = True

    def __call__(self, pcd: PointCloud) -> PointCloud:
        if random.random() > self.p:
            return pcd

        xyz, R, T, device = pcd.xyz, pcd.R, pcd.T, pcd.device

        if self.pair:
            if self.flag:
                x, y, z = (torch.rand(size=(3, )) - 0.5) * 2 * torch.pi  
            else:
                x, y, z = (torch.rand(size=(3, )) - 0.5) * 2 * self.r_std  
            x, y = x / 10, y / 10

            R_x = torch.tensor([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
            R_y = torch.tensor([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
            R_z = torch.tensor([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
            R_aug = R_x @ R_y @ R_z

            if self.flag:
                self.random_R = R_aug 
            else:
                R_aug = R_aug @ self.random_R 
            self.flag = not self.flag

        else:
            x, y, z = (torch.rand(size=(3,)) - 0.5) * 2 * self.r_std 
            x, y = x / 10, y / 10 
            R_x = torch.tensor([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
            R_y = torch.tensor([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
            R_z = torch.tensor([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
            R_aug = R_x @ R_y @ R_z

        R_aug = R_aug.to(pcd.device)

        if self.t_std > 0:
            T_aug = torch.normal(size=(3, 1), mean=self.t_mean, std=self.t_std, device=device)
            T_aug[2] /= 2
        else:
            T_aug = torch.zeros(size=(3, 1), device=device)

        pcd.xyz = (R_aug @ xyz.T + T_aug).T
        if pcd.has_norm:
            pcd.norm = (R_aug @ pcd.norm.T).T
        '''
        R @ pcd + T = R_new @ (R_aug @ pcd + T_aug) + T_new
                    = (R_new @ R_aug) @ pcd + R_new @ T_aug + T_new
        R = R_new @ R_aug
        T = R_new @ T_aug + T_new
        '''
        R_new = R @ R_aug.T
        T_new = T - R_new @ T_aug
        calib_SE3 = torch.eye(4, dtype=torch.float32, device=device)
        calib_SE3[:3, :3] = R_aug
        calib_SE3[:3, 3:] = T_aug
        pcd.calib = calib_SE3 @ pcd.calib

        pcd.R = R_new
        pcd.T = T_new
        return pcd


class RandomPosJitter:

    def __init__(self, mean: float = 0, std: float = 0.05, p: float = 1.0):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, pcd: PointCloud) -> PointCloud:
        if random.random() > self.p:
            return pcd

        pos_jitter = torch.normal(size=(pcd.nbr_point, 3), mean=self.mean, std=self.std, device=pcd.device)\
            .clamp(min=-3 * self.std, max=3 * self.std)
        pcd.xyz += pos_jitter
        return pcd


class ToGPU:

    def __init__(self):
        self.has_gpu = torch.cuda.is_available()
        # assert torch.cuda.is_available(), '\'ToGPU\' needs CUDA gpu, but CUDA is not available'

    def __call__(self, pcd: PointCloud) -> PointCloud:
        if (self.has_gpu):
            pcd.to_gpu()
        return pcd


class ToCPU:

    def __init__(self):
        pass

    def __call__(self, pcd: PointCloud) -> PointCloud:
        pcd.to_cpu()
        return pcd


class ToTensor:

    def __init__(self, use_norm: bool = False, use_uvd: bool = False, padding_to: int = -1, use_image: bool = False,
                 use_calib: bool = False):
        self.use_norm = use_norm
        self.use_uvd = use_uvd
        self.padding_to = padding_to
        self.use_image = use_image
        self.use_calib = use_calib

    def __call__(self, pcd: PointCloud):
        return pcd.to_tensor(use_norm=self.use_norm, use_uvd=self.use_uvd, padding_to=self.padding_to,
                             use_image=self.use_image, use_calib=self.use_calib)


pointcloud_transforms = {
    'GroundFilter': GroundFilter,
    'OutlierFilter': OutlierFilter,
    'LowPassFilter': LowPassFilter,
    'VerticalCorrect': VerticalCorrect,
    'VoxelSample': VoxelSample,
    'FarthestPointSample': FarthestPointSample,
    'RandomSample': RandomSample,
    'DistanceSample': DistanceSample,
    'CoordinatesNormalization': CoordinatesNormalization,
    'RandomShuffle': RandomShuffle,
    'RandomDrop': RandomDrop,
    'RandomShield': RandomOcclusion,
    'RandomRT': RandomRT,
    'RandomPosJitter': RandomPosJitter,
    'ToGPU': ToGPU,
    'ToCPU': ToCPU,
    'ToTensor': ToTensor
}


def get_transforms(args_dict: dict, return_list: bool = False) -> Union[Compose, List]:
    transforms_list = []
    for key, value in args_dict.items():
        if key != 'RandomChoice':
            transforms_list.append(pointcloud_transforms[key](**value))
        else:
            sub_transforms = get_transforms(value['transforms'], return_list=True)
            p = value['p']
            transforms_list.append(RandomChoice(transforms=sub_transforms, p=p))
    if return_list:
        return transforms_list
    else:
        return Compose(transforms=transforms_list)


class PointCloudTransforms:

    def __init__(self, args, mode: Literal['train', 'infer'] = 'train'):
        assert mode in ['train', 'infer']
        self.args = args
        self.transforms = get_transforms(args.transforms)
        self.mode = mode
        if mode == 'train':
            self._call_method = self._call_train
        else:
            self._call_method = self._call_infer

    def __call__(self, pcd: PointCloud):
        return self._call_method(pcd)

    def _call_train(self, pcd: PointCloud):
        return self.transforms(pcd)

    def _call_infer(self, pcd: PointCloud):
        original_pcd = pcd.xyz.clone()
        results = self.transforms(pcd)
        return *results, original_pcd
