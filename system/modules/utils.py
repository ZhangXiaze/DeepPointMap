import numpy as np
import open3d as o3d
import torch
from enum import Enum, unique
from queue import Queue
from typing import Any, Dict, List, Literal, Set, Tuple, Union
from matplotlib import pyplot as plt

try:
    from pytorch3d.ops.knn import knn_points
    has_torch3d = True
except:
    has_torch3d = False

agent_color = lambda agentid: plt.get_cmap('tab20')(2 * agentid + 1)[:3]
agent_color_darker = lambda agentid: tuple(i / 2 for i in plt.get_cmap('tab20')(2 * agentid)[:3])
coorsys_color = lambda coorid: plt.get_cmap('tab20')(2 * coorid)[:3]
simvec_to_num = lambda sim_vec: sim_vec.flatten()[:30].mean().item()


@unique
class EXIT_CODE(Enum):
    acpt = 0
    drop = 10
    dist = 11
    engy = 12
    exit = 21


class PoseTool(object):
    @classmethod
    def SE3(cls, R, t):
        if (isinstance(R, np.ndarray)):
            R = torch.tensor(R, dtype=torch.float32).reshape(3, 3)
        if (isinstance(t, np.ndarray)):
            t = torch.tensor(t, dtype=torch.float32).reshape(3, 1)
        mat = torch.eye(4)
        mat[:3, :3] = R
        mat[:3, 3:4] = t
        return mat

    @classmethod
    def Rt(cls, SE3):
        '''
        R: torch.Tensor(3, 3)
        t: torch.Tensor(3, 1)
        '''
        R = SE3[:3, :3]
        t = SE3[:3, 3:]
        return (R, t)

    @classmethod
    def rotation_angle(cls, rot_mat) -> float:
        '''
        rot_mat: torch.Tensor(3,3)
        '''
        return torch.arccos((torch.trace(rot_mat) - 1) / 2).item()


def calculate_information_matrix_from_pcd(pointcloud_1: torch.Tensor, pointcloud_2: torch.Tensor, SE3: torch.Tensor, device='cpu'):
    """Calculate information matrix for global optim

    Args:
        pointcloud_1 (torch.Tensor): source point cloud (3, N) float
        pointcloud_2 (torch.Tensor): target point cloud (3, N) float
        SE3 (torch.Tensor): estimate transformation matrix (4, 4) float

    Returns:
        torch.Tensor: information mat
    """
    if (has_torch3d):
        radius = 1.0
        R, T = PoseTool.Rt(SE3)
        R = R.to(device)
        T = T.to(device)
        pcd1, pcd2 = pointcloud_1.to(device), pointcloud_2.to(device)
        p1 = (R @ pcd1 + T).T.unsqueeze(0)
        p2 = pcd2.T.unsqueeze(0)

        result = knn_points(p1=p1, p2=p2, K=1, return_nn=False, return_sorted=False)
        idx = result.idx.squeeze(0).squeeze(-1)
        dists = result.dists.squeeze(0).squeeze(-1)
        dists_mask = dists <= (radius**2)
        corres = idx[dists_mask]

        t = pcd2[:, corres].T
        x, y, z = t[:, 0], t[:, 1], t[:, 2]
        GTG = torch.zeros(size=(6, 6), device=device)
        G_r = torch.zeros(size=(t.shape[0], 6, 1), device=device)
        G_r[:, 1, 0] = z
        G_r[:, 2, 0] = -y
        G_r[:, 3, 0] = 1.0
        GTG += (G_r @ G_r.transpose(1, 2)).sum(0)
        G_r.zero_()
        G_r[:, 0, 0] = -z
        G_r[:, 2, 0] = x
        G_r[:, 4, 0] = 1.0
        GTG += (G_r @ G_r.transpose(1, 2)).sum(0)
        G_r.zero_()
        G_r[:, 0, 0] = y
        G_r[:, 1, 0] = -x
        G_r[:, 5, 0] = 1.0
        GTG += (G_r @ G_r.transpose(1, 2)).sum(0)
        information_mat = GTG.cpu()
    else:
        info_mat_source = o3d.geometry.PointCloud()
        info_mat_source.points = o3d.open3d.utility.Vector3dVector(pointcloud_1.numpy().T)
        info_mat_target = o3d.geometry.PointCloud()
        info_mat_target.points = o3d.open3d.utility.Vector3dVector(pointcloud_2.numpy().T)
        information_mat = o3d.pipelines.registration.get_information_matrix_from_point_clouds(info_mat_source, info_mat_target, 1.0, SE3)
        information_mat = torch.tensor(information_mat, dtype=torch.float32)

    return information_mat


class Communicate_Module(object):
    OPERATIONS_type = Literal['NO_OP', 'UPLOAD_SCAN', 'AGENT_QUIT', 'QUIT']
    OPERATIONS = ['NO_OP', 'UPLOAD_SCAN', 'AGENT_QUIT', 'QUIT']

    def __init__(self) -> None:
        self.init = False
        self.agents: Set[int] = set()
        self.agent_queues: Dict[int, Queue[Tuple[Communicate_Module.OPERATIONS_type, Any]]] = dict()

    def add_member(self, system_id: int) -> None:
        self.agents |= {system_id}
        self.agent_queues |= {system_id: Queue()}
        self.logger = []

    def remove_member(self, system_id) -> None:
        self.agents.remove(system_id)
        del self.agent_queues[system_id]

    def get_members(self):
        return list(self.agents)

    def send_message(self, caller: int, callee: int, command: OPERATIONS_type, message: Any):
        assert command in self.OPERATIONS
        assert caller in self.agent_queues.keys()
        assert callee in self.agent_queues.keys()
        self.logger.append((caller, callee, command, message))
        self.agent_queues[callee].put((command, message))

    def fetch_message(self, system_id, block=True):
        if block:
            return self.agent_queues[system_id].get()
        else:
            if self.agent_queues[system_id].empty():
                return ('NO_OP', None)
            else:
                return self.agent_queues[system_id].get()

    def get_queue_length(self, system_id):
        return self.agent_queues[system_id].qsize()
