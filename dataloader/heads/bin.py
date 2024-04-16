import os
import numpy as np
from dataloader.heads.auto import PointCloudReader


class BinReader(PointCloudReader):
    optional_type = ['bin']

    def __init__(self):
        super().__init__()

    def _load_pcd(self, file_path):
        file_type = os.path.splitext(file_path)[-1][1:]
        assert file_type in self.optional_type, f'Only type of the file in {self.optional_type} is optional, ' \
                                                f'not \'{file_type}\''
        xyz = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)[:, :3]  # (N, 3)
        xyz = xyz[np.isnan(xyz).sum(1) == 0]
        rotation = None
        translation = None
        norm = None
        label = None
        image = None
        uvd = None

        return xyz, rotation, translation, norm, label, image, uvd

