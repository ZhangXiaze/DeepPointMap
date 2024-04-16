import os
import numpy as np
from dataloader.transforms import PointCloud


class PointCloudReader:
    optional_type = ['npz', 'npy', 'bin']

    def __init__(self):
        pass

    def __call__(self, file_path: str) -> PointCloud:

        xyz, rotation, translation, norm, label, image, uvd = self._load_pcd(file_path)
        pcd = PointCloud(xyz=xyz, rotation=rotation, translation=translation,
                         norm=norm, label=label, image=image, uvd=uvd)
        return pcd

    def _load_pcd(self, file_path):
        file_type = os.path.splitext(file_path)[-1][1:]
        assert file_type in self.optional_type, f'Only type of the file in {self.optional_type} is optional, ' \
                                                f'not \'{file_type}\''
        if file_type == 'npy':
            xyz = np.load(file_path)  # (N, 3)
            rotation = None
            translation = None
            norm = None
            label = None
            image = None
            uvd = None
        elif file_type == 'npz':
            with np.load(file_path, allow_pickle=True) as npz:
                npz_keys = npz.files
                assert 'lidar_pcd' in npz_keys, 'pcd file must contains \'lidar_pcd\''
                xyz = npz['lidar_pcd']  # (N, 3), f32
                rotation = npz['ego_rotation'] if 'ego_rotation' in npz_keys else None  # (3, 3), f32
                translation = npz['ego_translation'] if 'ego_translation' in npz_keys else None  # (3, 1), f32
                norm = npz['lidar_norm'] if 'lidar_norm' in npz_keys else None  # (N, 3), f32
                label = npz['lidar_seg'] if 'lidar_seg' in npz_keys else None  # (N, 3), f32
                image = None
                uvd = None
        elif file_type == 'bin':
            xyz = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]  # (N, 3)
            rotation = None
            translation = None
            norm = None
            label = None
            image = None
            uvd = None
        else:
            raise ValueError

        return xyz, rotation, translation, norm, label, image, uvd

