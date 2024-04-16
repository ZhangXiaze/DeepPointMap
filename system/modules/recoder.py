from typing import Dict, Union
import colorlog as logging
from os.path import join as ospj
import matplotlib.pyplot as plt
import numpy as np
import torch

import time
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import open3d as o3d

from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from system.modules.utils import PoseTool, agent_color, agent_color_darker
from system.modules.pose_graph import PoseGraph, ScanPack, PoseGraph_Edge


class ResultLogger():
    def __init__(self, args, system_info, posegraph_map: PoseGraph, log_dir: str) -> None:
        """_summary_

        Args:
            args (EasyDict):
                None
            system_info (EasyDict):
                None
            posegraph_map (PoseGraph): PoseGraph of SLAM system
            dpm_encoder (nn.Module): DPM Encoder Model
            device (str, optional): Inference Device. Defaults to 'cpu'.
        """
        self.args = args
        self.system_info = system_info
        self.log_dir = log_dir
        self.posegraph_map = posegraph_map

        self.time_recorder = dict()

    def interp_pose(self, timestamp: float):
        # [(time, Pos(3, 1))]
        current_pose_list = sorted([(s.timestamp, PoseTool.Rt(s.SE3_pred)[1]) for s in filter(lambda s: s.SE3_pred is not None, self.posegraph_map.get_all_scans())],
                                   key=lambda time_pose: time_pose[0])[-3:]
        xs, ys = zip(*current_pose_list)
        start_time = time.perf_counter()
        Pose_translation_interp = CubicSpline(xs, torch.stack(ys, dim=0).cpu().numpy(), axis=0)
        trans_interp = Pose_translation_interp([timestamp])[0]
        time_cost = time.perf_counter() - start_time
        if (time_cost * 1000 > 10):  # > 10ms
            logger.warning(f'translation interpolate took {time_cost*1000:.2f} ms, which is too slow...')
        return trans_interp

    def record_perf(self, name: str, time_s: float):
        if (name not in self.time_recorder.keys()):
            self.time_recorder[name] = list()
        self.time_recorder[name].append(time_s)

    def log_time(self, window: Union[None, int] = None) -> dict:
        ret = dict()
        for name, timelist in self.time_recorder.items():
            if (window is None):
                t = [t for t in timelist if t >0.0] 
            else:
                t = timelist[-window:] if window < len(timelist) else timelist
            ret[name] = (sum(t) / len(t),np.std(t))
        return ret

    def get_time_list(self, log_name: str):
        assert (log_name in self.time_recorder.keys())
        return self.time_recorder[log_name].copy()

    def save_trajectory(self, file_name: str = 'traj_kitti'):
        scans = list(self.posegraph_map.get_all_scans())
        scans.sort(key=lambda s: s.timestep)
        logger_file = open(ospj(self.log_dir, file_name + f'.allframes.txt'), 'w+')
        for s in scans:
            logger_file.write(' '.join([f'{i:.10f}' for i in s.SE3_pred[:3, :].flatten().tolist()]) + '\n')
        logger_file.close()

        logger_step_file = ospj(self.log_dir, file_name + f'.allsteps.txt')
        with open(logger_step_file, 'w+') as f:
            for s in scans:
                f.write(f'{int(s.timestep)}\n')

        logger_file = open(ospj(self.log_dir, file_name + f'.keyframes.txt'), 'w+')
        for s in filter(lambda s: s.type == 'full', scans):
            logger_file.write(' '.join([f'{i:.10f}' for i in s.SE3_pred[:3, :].flatten().tolist()]) + '\n')
        logger_file.close()

        logger_step_file = ospj(self.log_dir, file_name + f'.keysteps.txt')
        with open(logger_step_file, 'w+') as f:
            for s in filter(lambda s: s.type == 'full', scans):
                f.write(f'{int(s.timestep)}\n')

    def draw_trajectory(self, file_name: str = 'traj_jpg', draft: bool = False):
        PLOT_FACE_COLOR = (0.075, 0.075, 0.075, 1)
        if (draft):
            plt.figure(figsize=(10, 10), facecolor=PLOT_FACE_COLOR)
            ax = plt.axes()
            ax.set_rasterization_zorder(z=None)
        else:
            plt.figure(figsize=(20, 20), dpi=300, facecolor=PLOT_FACE_COLOR)
            ax = plt.axes()

        ax.axis('equal')
        ax.set_facecolor(PLOT_FACE_COLOR)

        # Plot scan
        scans = list(self.posegraph_map.get_all_scans())
        scans.sort(key=lambda s: s.timestep)
        for s in scans:
            if s.type == 'full':
                ax.plot(
                    s.SE3_pred[0, 3],
                    s.SE3_pred[1, 3],
                    color=agent_color(s.agent_id),
                    markersize=5,
                    linestyle='',
                    marker='o',
                    markeredgewidth=1,
                    markeredgecolor=agent_color_darker(s.agent_id),
                    zorder=10,
                )
            elif s.type == 'non-keyframe':
                ax.plot(
                    s.SE3_pred[0, 3],
                    s.SE3_pred[1, 3],
                    color=agent_color(s.agent_id),
                    markersize=5,
                    linestyle='',
                    marker=',',
                    alpha=0.3,
                    zorder=10,
                )
            ax.scatter(s.SE3_gt[0, 3], s.SE3_gt[1, 3], marker='.', c='white', zorder=9)

        # Plot edges
        for e in self.posegraph_map.get_all_edges():
            src_scan = self.posegraph_map.get_scanpack(e.src_scan_token)
            dst_scan = self.posegraph_map.get_scanpack(e.dst_scan_token)
            if (e.type == 'locz'):
                ax.plot([src_scan.SE3_pred[0, 3], dst_scan.SE3_pred[0, 3]], [src_scan.SE3_pred[1, 3], dst_scan.SE3_pred[1, 3]], color='lime', alpha=0.5, zorder=8)
            elif (e.type == 'loop'):
                ax.plot([src_scan.SE3_pred[0, 3], dst_scan.SE3_pred[0, 3]], [src_scan.SE3_pred[1, 3], dst_scan.SE3_pred[1, 3]], color='yellow', alpha=0.75, zorder=20)
            elif (e.type == 'odom'):
                ax.plot([src_scan.SE3_pred[0, 3], dst_scan.SE3_pred[0, 3]], [src_scan.SE3_pred[1, 3], dst_scan.SE3_pred[1, 3]], color='cyan', alpha=0.75, zorder=8)
            elif (e.type == 'prxy'):
                ax.plot([src_scan.SE3_pred[0, 3], dst_scan.SE3_pred[0, 3]], [src_scan.SE3_pred[1, 3], dst_scan.SE3_pred[1, 3]], color='purple', alpha=0.75, zorder=8)
        # Draft mode only draw pose-graph
        if (draft):
            plt.tight_layout()
            plt.savefig(ospj(self.log_dir, file_name + f'.map.jpg'))
            plt.close()
            return

        # Non-Draft mode draw pointclouds
        global_fullcloud_list = []
        global_keypoint_list = []
        view_R, view_T = torch.eye(3), torch.zeros(size=(3, 1))

        for scan in self.posegraph_map.get_all_scans():
            R_global, T_global = PoseTool.Rt(scan.SE3_pred)
            if (scan.full_pcd is not None):
                full_points = scan.full_pcd.clone()
                full_points[:3, :] = view_R @ (R_global @ (scan.full_pcd[:3, :]) + T_global) + view_T  # (xyz+others, N)
                global_fullcloud_list.append(full_points)
            if (scan.key_points is not None):
                key_points = scan.key_points.clone()
                key_points[-3:, :] = view_R @ (R_global @ (scan.key_points[-3:, :]) + T_global) + view_T  # (fea+xyz, N)
                global_keypoint_list.append(key_points)

        if (len(global_fullcloud_list) > 0):
            global_map_full = torch.cat(global_fullcloud_list, dim=1) if (len(global_fullcloud_list) > 0) else None  # (xyz+others, N)
            global_map_full_o3d = o3d.geometry.PointCloud()
            global_map_full_o3d.points = o3d.open3d.utility.Vector3dVector(global_map_full[:3, :].cpu().numpy().T)
            global_map_full_o3d_down = global_map_full_o3d.voxel_down_sample(voxel_size=0.5)
            global_map_full_xyz_down = np.asarray(global_map_full_o3d_down.points).T  # [xyz, N]
            del global_map_full_o3d_down
        else:
            global_map_full = global_map_full_o3d = global_map_full_xyz_down = None

        if (len(global_keypoint_list) > 0):
            global_map_key = torch.cat(global_keypoint_list, dim=1) if (len(global_keypoint_list) > 0) else None  # (fea+xyz, N)
            global_map_key_o3d = o3d.geometry.PointCloud()
            global_map_key_o3d.points = o3d.open3d.utility.Vector3dVector(global_map_key[-3:, :].cpu().numpy().T)
            global_map_key_o3d_down = global_map_key_o3d.voxel_down_sample(voxel_size=0.5)
            global_map_key_xyz_down = np.asarray(global_map_key_o3d_down.points).T  # [xyz, N]
            del global_map_key_o3d_down
        else:
            global_map_key = global_map_key_o3d = global_map_key_xyz_down = None

        if (global_map_full_xyz_down is not None):
            ax.scatter(global_map_full_xyz_down[0, :], global_map_full_xyz_down[1, :], s=0.5, color=agent_color(self.posegraph_map.agent_id), alpha=0.25, zorder=4)
        if (global_map_key_xyz_down is not None):
            ax.scatter(global_map_key_xyz_down[0, :], global_map_key_xyz_down[1, :], s=1, color=agent_color_darker(self.posegraph_map.agent_id), alpha=0.5, zorder=5)

        plt.tight_layout()
        plt.savefig(ospj(self.log_dir, file_name + f'.map.jpg'))
        plt.close()

    def save_map(self, file_name: str = 'map'):
        global_fullcloud_list = []
        global_keypoint_list = []
        xyz_pred = []
        for scan in self.posegraph_map.get_all_scans():
            R_global, T_global = PoseTool.Rt(scan.SE3_pred)
            xyz_pred.append(T_global.flatten().tolist())
            if (scan.full_pcd is not None):
                full_points = scan.full_pcd.clone()
                full_points[:3, :] = R_global @ (scan.full_pcd[:3, :]) + T_global  # (xyz+others, N)
                global_fullcloud_list.append(full_points)
            if (scan.key_points is not None):
                key_points = scan.key_points.clone()
                key_points[-3:, :] = R_global @ (scan.key_points[-3:, :]) + T_global  # (fea+xyz, N)
                global_keypoint_list.append(key_points)

        # if (len(global_fullcloud_list) > 0):
        #     global_map_full = torch.cat(global_fullcloud_list, dim=1) if (len(global_fullcloud_list) > 0) else None  # (xyz+others, N)
        #     global_map_full_o3d = o3d.geometry.PointCloud()
        #     global_map_full_o3d.points = o3d.open3d.utility.Vector3dVector(global_map_full[:3, :].cpu().numpy().T)
        #     o3d.io.write_point_cloud(ospj(self.log_dir, file_name + f'.fullpoints.pcd'), global_map_full_o3d)
        # del global_fullcloud_list
        #
        # if (len(global_keypoint_list) > 0):
        #     global_map_key = torch.cat(global_keypoint_list, dim=1) if (len(global_keypoint_list) > 0) else None  # (fea+xyz, N)
        #     global_map_key_o3d = o3d.geometry.PointCloud()
        #     global_map_key_o3d.points = o3d.open3d.utility.Vector3dVector(global_map_key[-3:, :].cpu().numpy().T)
        #     o3d.io.write_point_cloud(ospj(self.log_dir, file_name + f'.keypoints.pcd'), global_map_key_o3d)
        # del global_keypoint_list
        #
        # lines_pcd = o3d.geometry.LineSet()
        # lines_pcd.lines = o3d.utility.Vector2iVector([i, i + 1] for i in range(len(xyz_pred) - 1))
        # lines_pcd.points = o3d.utility.Vector3dVector(torch.tensor(xyz_pred).numpy())
        # lines_pcd.paint_uniform_color((0, 1, 0))
        # o3d.io.write_line_set(ospj(self.log_dir, file_name + f'.traj.ply'), lines_pcd)

    def save_posegraph(self, file_name: str = 'posegraph'):
        g2o_file_path = ospj(self.log_dir, file_name + f'.pg.g2o')
        self.posegraph_map.to_g2o_file(g2o_file_path)
