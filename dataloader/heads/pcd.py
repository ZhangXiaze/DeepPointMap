import os
import open3d
import numpy as np
from dataloader.heads.auto import PointCloudReader


class PcdReader(PointCloudReader):
    optional_type = ['pcd']

    def __init__(self):
        super().__init__()

    def _load_pcd(self, file_path):
        file_type = os.path.splitext(file_path)[-1][1:]
        assert file_type in self.optional_type, f'Only type of the file in {self.optional_type} is optional, ' \
                                                f'not \'{file_type}\''
        pcd = open3d.io.read_point_cloud(file_path)
        xyz = np.asarray(pcd.points)
        xyz = xyz[np.sum(np.isnan(xyz), axis=-1) == 0]
        rotation = None
        translation = None
        norm = None
        label = None
        image = None
        uvd = None

        return xyz, rotation, translation, norm, label, image, uvd

