import os
import numpy as np
from dataloader.heads.auto import PointCloudReader


class NPZReader(PointCloudReader):
    optional_type = ['npz']

    def __init__(self):
        super().__init__()

    def _load_pcd(self, file_path):
        file_type = os.path.splitext(file_path)[-1][1:]
        assert file_type in self.optional_type, f'Only type of the file in {self.optional_type} is optional, ' \
                                                f'not \'{file_type}\''
        with np.load(file_path, allow_pickle=True) as npz:
            npz_keys = npz.files
            assert 'lidar_pcd' in npz_keys, 'pcd file must contains \'lidar_pcd\''
            xyz = npz['lidar_pcd']  # (N, 3), f32
            rotation = npz['ego_rotation'] if 'ego_rotation' in npz_keys else None  # (3, 3), f32
            translation = npz['ego_translation'] if 'ego_translation' in npz_keys else None  # (3, 1), f32
            norm = npz['lidar_norm'] if 'lidar_norm' in npz_keys else None  # (N, 3), f32
            label = npz['lidar_seg'] if 'lidar_seg' in npz_keys else None  # (N, 3), f32
            image = npz['image'] if 'image' in npz_keys else None
            uvd = npz['lidar_proj'] if 'lidar_proj' in npz_keys else None

        return xyz, rotation, translation, norm, label, image, uvd

