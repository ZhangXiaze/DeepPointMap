import colorlog as logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Union
import time

from system.modules.utils import PoseTool, calculate_information_matrix_from_pcd

from system.modules.pose_graph import PoseGraph, ScanPack, PoseGraph_Edge
from system.modules.utils import simvec_to_num


class ExtractionThread():
    def __init__(self, args, system_info, posegraph_map: PoseGraph, dpm_encoder: nn.Module, device='cpu') -> None:
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
        self.device = device
        self.encoder_model = dpm_encoder.to(self.device).eval()
        self.posegraph_map = posegraph_map

    def process(self, point_cloud: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        start_t = time.perf_counter()
        with torch.no_grad():
            descriptors_xyz, descriptors_fea, mask = self.encoder_model(point_cloud.to(self.device), padding_mask.to(self.device))
        # de-padding, [128, N], [3, N] and scale to real size (meter)

        # descriptors_fea = descriptors_fea[0, :, ~mask[0]]
        # descriptors_xyz = descriptors_xyz[0, :, ~mask[0]] * self.args.coor_scale
        net_t = time.perf_counter()

        descriptors_fea = descriptors_fea
        descriptors_xyz = descriptors_xyz * self.args.coor_scale

        descriptors = torch.cat([descriptors_fea, descriptors_xyz], dim=1)  # [B, 128 + 3, N]
        end_t = time.perf_counter()

        logger.info(f'Agent{self.system_info.agent_id}: Extract log: net = {net_t - start_t:.4f}s, cat = {end_t - net_t:.4f}s')
        logger.info(f'Agent{self.system_info.agent_id}: Extract done, point = {point_cloud.shape[-1]} descriptor = {descriptors.shape[-1]}')
        return descriptors  # B, 128+3, N


class OdometryThread():
    def __init__(self, args, system_info, posegraph_map: PoseGraph, dpm_decoder: nn.Module, device='cpu') -> None:
        """

        Args:
            args (EasyDict):
                odometer_candidates_num (int): odometer candidates number 
                registration_sample_odometer (float | int): Registration sample number / ratio
            system_info (EasyDict):
            posegraph_map (PoseGraph): PoseGraph of SLAM system
            dpm_decoder (nn.Module): DPM Decoder Model
            device (str, optional): Inference Device. Defaults to 'cpu'.
        """
        self.args = args
        self.system_info = system_info
        self.posegraph_map = posegraph_map
        self.device = device
        self.decoder_model = dpm_decoder.to(self.device).eval()

    def search_candidates(self, new_scan: ScanPack) -> List[ScanPack]:

        if (len(self.posegraph_map.get_all_scans()) == 0 or new_scan.agent_id not in [s.agent_id for s in self.posegraph_map.get_all_scans()] or self.posegraph_map.last_known_keyframe is None
                or self.posegraph_map.last_known_anyframe is None):
            return []  # first scan in pose graph

        last_scan = self.posegraph_map.get_scanpack(self.posegraph_map.last_known_keyframe)
        last_SE3 = self.posegraph_map.get_scanpack(self.posegraph_map.last_known_anyframe).SE3_pred  # if self.last_known_position is not None else last_scan.SE3_pred

        assert last_SE3 is not None
        assert last_scan is not new_scan, f"last_scan is new_scan, call _add_odometry() BEFORE self.posegraph_map.add_scan()"

        key_frames = list(
            filter(lambda s: (s.type != 'non-keyframe' and s.agent_id == new_scan.agent_id),
                   self.posegraph_map.graph_search(token=last_scan.token, neighbor_level=5, coor_sys=last_scan.coor_sys, edge_type=['odom', 'loop'])))

        assert set([s.SE3_pred is not None for s in key_frames]) == {True}

        key_frame_distances = torch.norm(torch.stack([s.SE3_pred[:3, 3:] for s in key_frames], dim=0) - last_SE3[:3, 3:], p=2, dim=1)  # (N, 1)
        topk_key_frame_dist, topk_keyframe_index = torch.topk(key_frame_distances, dim=0, k=min(len(key_frames), self.args.odometer_candidates_num), largest=False)
        topk_keyframe_index: List[int] = topk_keyframe_index.flatten().tolist()
        key_frames = [key_frames[i] for i in topk_keyframe_index]
        if (key_frame_distances.min() > 20):
            logger.warning(f'The nearest key-frame seems too far ({key_frame_distances.min():.3f}m)')

        return key_frames

    def odometry(self, new_scan: ScanPack, candidates: List[ScanPack]) -> List[PoseGraph_Edge]:
        edges: List[PoseGraph_Edge] = []
        for nearest_scan in candidates:
            # Registration
            with torch.no_grad():
                rot_pred, trans_pred, sim_topks, rmse = self.decoder_model.registration_forward(nearest_scan.key_points.to(self.device),
                                                                                                new_scan.key_points.to(self.device),
                                                                                                num_sample=self.args.registration_sample_odometer)
            SE3 = PoseTool.SE3(rot_pred, trans_pred)

            # Calculate InfoMat

            information_mat = calculate_information_matrix_from_pcd(nearest_scan.full_pcd[:3, :], new_scan.full_pcd[:3, :], SE3, device=self.device)
            # information_mat = calculate_information_matrix_from_confidence(confidence=simvec_to_num(sim_topks), rmse=rmse)

            # Make Edges
            edge = PoseGraph_Edge(src_scan_token=nearest_scan.token,
                                  dst_scan_token=new_scan.token,
                                  SE3=SE3.inverse(),
                                  information_mat=information_mat,
                                  type='odom',
                                  confidence=simvec_to_num(sim_topks),
                                  rmse=rmse)
            edges.append(edge)
        return edges

    def process(self, new_scan: ScanPack) -> List[PoseGraph_Edge]:
        start_t = time.perf_counter()
        candidates = self.search_candidates(new_scan=new_scan)
        mid_t = time.perf_counter()
        edges = self.odometry(new_scan=new_scan, candidates=candidates)
        end_t = time.perf_counter()
        logger.info(f'Agent{self.system_info.agent_id}: Odometry log: search_candidates = {(mid_t - start_t)*1000:.4f}ms, odometry = {(end_t - mid_t)*1000:.4f}ms')
        return edges