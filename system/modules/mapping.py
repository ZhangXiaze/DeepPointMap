import colorlog as logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
import time
import torch
import torch.nn as nn
from typing import List, Tuple, Union

from system.modules.utils import PoseTool, EXIT_CODE, calculate_information_matrix_from_pcd
from system.modules.pose_graph import PoseGraph, ScanPack, PoseGraph_Edge
from system.modules.utils import simvec_to_num


class MappingThread():
    def __init__(self, args, system_info, posegraph_map: PoseGraph, dpm_decoder: nn.Module, device='cpu') -> None:
        """_summary_

        Args:
            args (EasyDict):
                key_frame_distance (float | auto): distance threshold for keyframe detection (auto = automatic distance adjustment).
                edge_confidence_drop (float): The minium confidence of odometer edge
                edge_rmse_drop (float): The maximum RMSE of odometer edge
                max_continuous_drop_scan (int): the maximum number of contiguous dropped scan. Auto revert/break will applied if exceeds.
                continuous_drop_scan_strategy ('recover' | 'break'): What to do if continuous dropped scans exceeds `max_continuous_drop_scan`
                enable_s2m_adjust (bool): Enable scan-to-map refinement
                registration_sample_mapping (float | int): Registration sample number / ratio
            system_info (EasyDict):
                agent_id (int): agent id
            posegraph_map (PoseGraph): PoseGraph of SLAM system
            dpm_decoder (nn.Module): DPM Decoder Model
            device (str, optional): Inference Device. Defaults to 'cpu'.
        """
        self.args = args
        self.system_info = system_info
        self.posegraph_map = posegraph_map
        self.device = device
        self.decoder_model = dpm_decoder.to(self.device).eval()

        self.dist_ratio = 1.0  # between [1.0, 2.0]
        if self.args.key_frame_distance == 'auto':
            self.dist_auto_adjust = True
            self.key_frame_distance_0 = getattr(self.args,'key_frame_distance_0',3.0)
            logger.info(f'{self.key_frame_distance_0=}')
            self.current_key_frame_distance = self.key_frame_distance_0 * self.dist_ratio
        else:
            self.dist_auto_adjust = False
            self.key_frame_distance_0 = self.args.key_frame_distance
            self.current_key_frame_distance = self.key_frame_distance_0

        self.drop_scans_bag: List[Tuple[ScanPack, PoseGraph_Edge]] = []

    def valid_check(self, new_scan: ScanPack, edge: PoseGraph_Edge):
        if (edge.confidence < self.args.edge_confidence_drop) or (edge.rmse > self.args.edge_rmse_drop):
            # too bad, drop
            self.drop_scans_bag.append((new_scan, edge))
            logger.info(
                f'system {self.system_info.agent_id}: Current scan ({new_scan.token}) with confidence {edge.confidence:.3f} < {self.args.edge_confidence_drop} or rmse {edge.rmse:.4f} > {self.args.edge_rmse_drop:.3f} has been dropped'
            )
            if len(self.drop_scans_bag) >= self.args.max_continuous_drop_scan:
                if self.args.continuous_drop_scan_strategy == 'recover':
                    new_scan, edge = min(self.drop_scans_bag, key=lambda x: x[1].rmse)
                    logger.info(f'system {self.system_info.agent_id}: Too many dropped scans, recover scan ({new_scan.token}) with confidence {edge.confidence:.2f} and rmse {edge.rmse:.4f}')
                    self.drop_scans_bag.clear()
                    return EXIT_CODE.acpt
                elif self.args.continuous_drop_scan_strategy == 'break':
                    old_scan = self.posegraph_map.get_scanpack(self.posegraph_map.last_known_anyframe)
                    new_scan.SE3_pred = old_scan.SE3_pred
                    new_scan.coor_sys = old_scan.coor_sys
                    self.posegraph_map.add_vertex(new_scan)
                    self.posegraph_map.last_known_keyframe = new_scan.token
                    self.posegraph_map.last_known_anyframe = new_scan.token
                    logger.info(f'system {self.system_info.agent_id}: Too many dropped scans, break posegraph ({new_scan.token})')
                    self.drop_scans_bag.clear()
                    return EXIT_CODE.acpt
                else:
                    raise ValueError
            else:
                return EXIT_CODE.drop
        else:
            self.drop_scans_bag.clear()
            return EXIT_CODE.acpt

    def keyframe_check(self, new_scan: ScanPack, edge: PoseGraph_Edge):
        if self.dist_auto_adjust:
            m = 0.90
            rmse_ratio = min(edge.rmse / self.args.edge_rmse_drop, 1.0)

            this_ratio = ((1 - rmse_ratio)**2) * 2.0
            self.dist_ratio = max(min(m * self.dist_ratio + (1 - m) * this_ratio, 2.0), 0.0)
            self.current_key_frame_distance = max(self.key_frame_distance_0 * self.dist_ratio, 1.0)

            # v2
            # this_ratio = (-(2 * rmse_ratio - 1) ** 3 + 1) / 2
            # self.dist_ratio = max(min(m * self.dist_ratio + (1 - m) * this_ratio, 1.0), 0.2)
            # self.current_key_frame_distance = 5 * self.dist_ratio

            # v2.1
            # this_ratio = 0.3 + ((1 - rmse_ratio) ** 2) * (2.0 - 0.3)
            # self.dist_ratio = m * self.dist_ratio + (1 - m) * this_ratio
            # self.current_key_frame_distance = self.key_frame_distance_0 * self.dist_ratio

            # v2.2
            # this_ratio = (1 - rmse_ratio) * 2.0
            # self.dist_ratio = max(min(m * self.dist_ratio + (1 - m) * this_ratio, 2.0), 0.3)
            # self.current_key_frame_distance = self.key_frame_distance_0 * self.dist_ratio

        '''Key Frame Verify'''
        old_scan = self.posegraph_map.get_scanpack(edge.src_scan_token)
        new_scan = new_scan
        assert old_scan.token == edge.src_scan_token
        assert new_scan.token == edge.dst_scan_token
        new_scan.SE3_pred = old_scan.SE3_pred @ edge.SE3
        new_scan.coor_sys = old_scan.coor_sys

        assert old_scan.type != 'non-keyframe'
        self.posegraph_map.last_known_keyframe = old_scan.token

        # Dist
        if self.current_key_frame_distance >= 0:
            xyz_of_querys = new_scan.SE3_pred[:3, 3].unsqueeze(0)  # (1, 2)
            global_map_scan_list = list(
                filter(lambda s: s.type != 'non-keyframe', self.posegraph_map.graph_search(token=old_scan.token, neighbor_level=5, coor_sys=new_scan.coor_sys,
                                                                                           edge_type=['odom', 'loop'])))  # [scan1, scan2, ...]
            xyz_of_scans = torch.stack([PoseTool.Rt(s.SE3_pred)[1][:, 0] for s in global_map_scan_list], dim=0)  # (N, 3)
            distance = torch.norm(xyz_of_scans - xyz_of_querys, p=2, dim=1).min()  # (N, ) -> (1, ), distance
            if distance < self.current_key_frame_distance:
                return EXIT_CODE.dist
        # Engy
        if False and self.args.key_frame_energy < 1:
            raise NotImplementedError
            # position_energy_graph_space = self.posegraph_map.global_energy_calculate_graph(SE3=new_scan.SE3_pred, token=old_scan.token, coor_sys=new_scan.coor_sys, neighbor_level=5).item()
            # if position_energy_graph_space > self.args.key_frame_energy:
            #     return EXIT_CODE.engy
        return EXIT_CODE.acpt

    def scan_to_map_adjustment(self, edge: PoseGraph_Edge):
        if not self.args.enable_s2m_adjust:
            return edge
        else:
            src_scan_old = self.posegraph_map.get_scanpack(edge.src_scan_token)
            dst_scan_new = self.posegraph_map.get_scanpack(edge.dst_scan_token)

            src_map_descriptors, src_map_src_map_descriptor_token = self.posegraph_map.global_map_query_graph(token=src_scan_old.token,
                                                                                                              neighbor_level=5,
                                                                                                              coor_sys=src_scan_old.coor_sys,
                                                                                                              full_pcd=False,
                                                                                                              centering_SE3=src_scan_old.SE3_pred)
            src_map_descriptors = src_map_descriptors[:, src_map_src_map_descriptor_token != dst_scan_new.token]  # drop same descriptors from map

            # Registration
            with torch.no_grad():
                rot_pred, trans_pred, sim_topks, rmse = \
                    self.decoder_model.registration_forward(src_map_descriptors.to(self.device),
                                                            dst_scan_new.key_points.to(self.device),
                                                            num_sample=self.args.registration_sample_mapping)
            SE3 = PoseTool.SE3(rot_pred, trans_pred)

            # Calculate InfoMat
            information_mat = calculate_information_matrix_from_pcd(src_scan_old.full_pcd[:3, :], dst_scan_new.full_pcd[:3, :], SE3, device=self.device)
            # information_mat = calculate_information_matrix_from_confidence(confidence=simvec_to_num(sim_topks), rmse=rmse)

            # Make Edges
            edge = PoseGraph_Edge(src_scan_token=edge.src_scan_token,
                                  dst_scan_token=edge.dst_scan_token,
                                  SE3=SE3.inverse(),
                                  information_mat=information_mat,
                                  type='odom',
                                  confidence=simvec_to_num(sim_topks),
                                  rmse=rmse)
            return edge

    def process(self, new_scan: ScanPack, odom_edge: PoseGraph_Edge) -> Union[EXIT_CODE, PoseGraph_Edge]:
        start_t = time.process_time()
        # Valid Check
        result = self.valid_check(new_scan=new_scan, edge=odom_edge)
        if result != EXIT_CODE.acpt:
            # edge type is drop
            # Do not add this edge into map
            # Do not add this vertex NOW
            return result
        else:
            self.posegraph_map.last_known_keyframe = odom_edge.src_scan_token

        # Keyframe Check
        result = self.keyframe_check(new_scan=new_scan, edge=odom_edge)
        if result != EXIT_CODE.acpt:
            self.posegraph_map.add_vertex(new_scan.nonkeyframe())
            self.posegraph_map.last_known_anyframe = new_scan.token
            odom_edge.type = 'locz'
            self.posegraph_map.add_edge(odom_edge)
            return result
        else:
            self.posegraph_map.add_vertex(new_scan.copy())
            self.posegraph_map.last_known_anyframe = new_scan.token
            self.posegraph_map.last_known_keyframe = new_scan.token
            odom_edge.type = 'odom'
            self.posegraph_map.add_edge(odom_edge)

        # Scan-to-Map Refinement
        odom_edge_new = self.scan_to_map_adjustment(odom_edge)
        if odom_edge_new.rmse <= self.args.edge_rmse_drop or odom_edge_new.rmse <= odom_edge.rmse:
            src_scan_old = self.posegraph_map.get_scanpack(odom_edge_new.src_scan_token)
            new_scan.SE3_pred = src_scan_old.SE3_pred @ odom_edge_new.SE3
            self.posegraph_map.update_scan_token(new_scan.token, new_SE3_pred=new_scan.SE3_pred)
            self.posegraph_map.update_edge_token(odom_edge.src_scan_token, odom_edge.dst_scan_token,
                                                 new_SE3=odom_edge_new.SE3, new_confidence=odom_edge_new.confidence,
                                                 new_information_mat=odom_edge_new.information_mat,
                                                 new_rmse=odom_edge_new.rmse)
        end_t = time.process_time()

        # Logger
        deltaT = torch.linalg.inv(odom_edge_new.SE3) @ odom_edge.SE3
        logger.info(f'Agent{self.system_info.agent_id}: Mapping log: {end_t - start_t:.4f}s')
        logger.info(f'Agent{self.system_info.agent_id}: |- M2M adjust: {torch.norm(deltaT[:3,3],p=2,dim=0).item():.3f} m {PoseTool.rotation_angle(deltaT[:3,:3])*180/torch.pi:.3f} deg')
        logger.info(f'Agent{self.system_info.agent_id}: Mapping done, add one keyframe {new_scan.token}')

        return odom_edge_new
