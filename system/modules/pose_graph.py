import time
from typing import Any, Callable, Dict, List, Literal, Set, Tuple, Union, Optional

import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation as sci_R
import torch
import colorlog as logging
from readerwriterlock import rwlock
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from system.modules.utils import PoseTool
from utils.device import detach_to_cpu


class ScanPack(object):
    """A structure that storage the information of a Scan

    """
    def __init__(
        self,
        timestamp: float,
        agent_id: int,
        timestep: int,
        key_points: torch.Tensor,
        scan_feature: Optional[torch.Tensor] = None,
        full_pcd: Optional[torch.Tensor] = None,
        SE3_pred: Optional[torch.Tensor] = None,
        SE3_gt: Optional[torch.Tensor] = None,
        gps_position: Optional[torch.Tensor] = None,
        fixed: bool = False,
        coor_sys: int = -1,
    ) -> None:
        self.token: int = (agent_id << 16) + timestep

        self.timestep: int = timestep  # int
        self.timestamp: float = timestamp  # float, in seconds
        self.agent_id: int = agent_id

        self.key_points: Optional[torch.Tensor] = detach_to_cpu(key_points)  # [128 + 3, 256] = [fea+xyz, N]
        self.scan_feature: Optional[torch.Tensor] = detach_to_cpu(scan_feature)  # [1, 1024]
        self.full_pcd: Optional[torch.Tensor] = detach_to_cpu(full_pcd)  # [xyz+norm, N]

        self.SE3_pred: Optional[torch.Tensor] = detach_to_cpu(SE3_pred).reshape(4, 4) if SE3_pred is not None else None
        self.SE3_gt: Optional[torch.Tensor] = detach_to_cpu(SE3_gt).reshape(4, 4) if SE3_gt is not None else None
        self.fixed: bool = fixed

        self.type: Literal['full', 'non-keyframe'] = 'full'
        self.coor_sys: int = coor_sys
        if gps_position is not None:
            self.gps_position = gps_position.view(3, 1)
        else:
            self.gps_position = torch.zeros(3, 1)

    def to(self, device) -> 'ScanPack':
        """Transfer this ScanPack to device (cpu, cuda or others...), INPLACE!

        Args:
            device (str): device of torch, e.g., 'cpu', 'cuda', etc...

        Returns:
            ScanPack: transferred ScanPack
        """
        self.key_points = self.key_points.to(device)
        self.full_pcd = self.full_pcd.to(device)
        return self

    def copy(self):
        '''
        deep copy with everything
        '''
        copy = ScanPack(timestamp=self.timestamp,
                        agent_id=self.agent_id,
                        timestep=self.timestep,
                        key_points=self.key_points,
                        scan_feature=self.scan_feature,
                        full_pcd=self.full_pcd,
                        SE3_pred=self.SE3_pred,
                        SE3_gt=self.SE3_gt,
                        gps_position=self.gps_position,
                        fixed=self.fixed,
                        coor_sys=self.coor_sys)
        return copy

    def nonkeyframe(self):
        '''
        deep copy without full-pcd and key-points
        '''
        copy = self.copy()
        copy.type = 'non-keyframe'

        # copy.full_pcd = None
        # assert (copy.full_pcd is None)

        copy.key_points = None
        assert (copy.key_points is None)
        return copy

    def __hash__(self) -> int:
        return self.token

    def __str__(self) -> str:
        return f'ScanPack {self.token}, type {self.type}'


class PoseGraph_Edge(object):
    """Edge in Pose Graph

    Note: The attr `SE3` represents the transformation of **agent** from src to dst

    Equally, src_points = R @ dst_points + t
    """
    def __init__(self, src_scan_token: int, dst_scan_token: int, SE3: torch.Tensor, information_mat: torch.Tensor, type: Literal['odom', 'loop', 'locz', 'prxy'], confidence=None, rmse=None) -> None:
        self.src_scan_token: int = src_scan_token
        self.dst_scan_token: int = dst_scan_token
        self.type: Literal['odom', 'loop', 'locz', 'prxy'] = type

        self.SE3: torch.Tensor = detach_to_cpu(SE3).reshape(4, 4)
        self.information_mat: torch.Tensor = detach_to_cpu(information_mat).reshape(6, 6)
        self.confidence = confidence
        self.rmse = rmse

    def to(self, device):
        return self

    def copy(self):
        copy = PoseGraph_Edge(src_scan_token=self.src_scan_token,
                              dst_scan_token=self.dst_scan_token,
                              SE3=self.SE3,
                              information_mat=self.information_mat,
                              type=self.type,
                              confidence=self.confidence,
                              rmse=self.rmse)
        return copy

    def __hash__(self) -> int:
        return (self.src_scan_token << 32) + self.dst_scan_token

    def __str__(self) -> str:
        return f'Edge {self.src_scan_token}<->{self.dst_scan_token}'


class PoseGraph(object):
    def __init__(self, args, agent_id: int, device: str) -> None:
        self.args = args
        self.device = device

        self.vertex: Dict[int, ScanPack] = dict()
        self.edge: Dict[Tuple[int, int], PoseGraph_Edge] = dict()

        self.key_frame_num = 0
        self.all_frame_num = 0

        self.odom_edge_num = 0
        self.loop_edge_num = 0
        self.locz_edge_num = 0
        self.prxy_edge_num = 0

        # scan_token -> [KeyPoints, FullPcds]
        self.__global_map_cache: Dict[int, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = dict()

        self.agent_id = agent_id
        self.last_known_keyframe: Union[int, None] = None
        self.last_known_anyframe: Union[int, None] = None
        self.uncertain = False

        self.optim = self.__optim_open3d
        self.locker = rwlock.RWLockFair()

    def add_vertex(self, scan: ScanPack, blocking=True):
        write_lock = self.locker.gen_wlock()
        write_lock.acquire(blocking=blocking)
        assert scan.token not in self.vertex.keys(), f'Scan {scan.token} already in posegraph map'
        self.vertex[scan.token] = scan
        self.__global_map_cache[scan.token] = [None, None]
        if (scan.type == 'full'):
            self.key_frame_num += 1
        self.all_frame_num += 1
        write_lock.release()

    def add_edge(self, new_edge: PoseGraph_Edge, blocking=True):
        if new_edge is None:
            return
        # __DEBUG__
        if self.has_scan(new_edge.src_scan_token) is False:
            logger.warning(f'Scan {new_edge.src_scan_token} not exists')
            raise RuntimeError(f'Scan {new_edge.dst_scan_token} not exists')
        # __DEBUG__
        if self.has_scan(new_edge.dst_scan_token) is False:
            logger.warning(f'Scan {new_edge.dst_scan_token} not exists')
            raise RuntimeError(f'Scan {new_edge.dst_scan_token} not exists')
        # __DEBUG__
        if ((new_edge.src_scan_token, new_edge.dst_scan_token) in self.edge.keys()):
            logger.warning(f'Received an edge that already exists ({new_edge.src_scan_token} - {new_edge.dst_scan_token})')
            raise RuntimeError(f'Received an edge that already exists ({new_edge.src_scan_token} - {new_edge.dst_scan_token})')
        # __DEBUG__
        if ((new_edge.dst_scan_token, new_edge.src_scan_token) in self.edge.keys()):
            logger.warning(f'Received an edge that already exists ({new_edge.dst_scan_token} - {new_edge.src_scan_token})')
            raise RuntimeError(f'Received an edge that already exists ({new_edge.dst_scan_token} - {new_edge.src_scan_token})')
        write_lock = self.locker.gen_wlock()
        write_lock.acquire(blocking=blocking)
        self.edge[(new_edge.src_scan_token, new_edge.dst_scan_token)] = new_edge
        # self.edge[(new_edge.dst_scan_token, new_edge.src_scan_token)] = new_edge.inv()
        if (new_edge.type == 'odom'):
            self.odom_edge_num += 1
        elif (new_edge.type == 'locz'):
            self.locz_edge_num += 1
        elif (new_edge.type == 'loop'):
            self.loop_edge_num += 1
        elif (new_edge.type == 'prxy'):
            self.prxy_edge_num += 1
        write_lock.release()

    def has_scan(self, scan_token):
        return scan_token in self.vertex.keys()

    def has_edge(self, src_token: int, end_token: int):
        exists = (src_token, end_token) in self.edge.keys()
        return exists

    @classmethod
    def get_agent_id(cls, token: int):
        return token >> 16

    def get_neighbor_tokens(self, scan_token, blocking=True) -> List[int]:
        """Get token of neighbors (r-Locked)

        Args:
            scan_token (int): Scan token

        Returns:
            List: Token lists
        """
        neighbors = []
        read_lock = self.locker.gen_rlock()
        read_lock.acquire(blocking=blocking)
        for src, dst in self.edge.keys():
            if (src == scan_token):
                neighbors.append(dst)
            elif (dst == scan_token):
                neighbors.append(src)
        read_lock.release()
        return neighbors

    def get_edge(self, src_scan_token, dst_scan_token):
        if ((src_scan_token, dst_scan_token) not in self.edge.keys()):
            if ((dst_scan_token, src_scan_token) not in self.edge.keys()):
                raise f'edge {(src_scan_token,dst_scan_token)} not exists'
            else:
                raise f'edge {(src_scan_token,dst_scan_token)} not exists. However, its reverse {(dst_scan_token,src_scan_token)} exists.'

        return self.edge[(src_scan_token, dst_scan_token)]

    def get_scanpack(self, scan_token: int) -> ScanPack:
        return self.vertex[scan_token]

    def update_scan_token(self, scan_token: int, new_SE3_pred=None, new_coor_sys=None, blocking=True):
        '''
        Given a new ScanPack, update the information within pose graph (w-Locked)

        only the agent_id, SE3_Pred, coor_sys and fixed can be modified
        '''

        # Update python dict
        write_lock = self.locker.gen_wlock()
        write_lock.acquire(blocking=blocking)
        scan = self.vertex[scan_token]
        if (new_SE3_pred is not None):
            scan.SE3_pred = new_SE3_pred  # update python SE3
            self.__global_map_cache[scan.token] = [None, None]
        if (new_coor_sys is not None):
            scan.coor_sys = new_coor_sys
        write_lock.release()

    def update_edge_token(self, src_token, dst_token, blocking=True, new_SE3=None, new_confidence=None, new_information_mat=None, new_rmse=None):
        if ((src_token, dst_token) not in self.edge.keys()):
            if ((dst_token, src_token) not in self.edge.keys()):
                raise f'edge [{src_token} -> {dst_token}] not exists'
            else:
                raise f'edge [{src_token} -> {dst_token}] not exists. However, its reverse [{dst_token} -> {src_token}] exists.'
        write_lock = self.locker.gen_wlock()
        write_lock.acquire(blocking=blocking)
        src2dst_edge = self.edge[(src_token, dst_token)]
        # dst2src_edge = self.edge[dst_token, src_token]
        if new_SE3 is not None:
            src2dst_edge.SE3 = new_SE3
            # dst2src_edge.SE3 = new_SE3.inverse()
        if new_confidence is not None:
            src2dst_edge.confidence = new_confidence
            # dst2src_edge.confidence = new_confidence
        if new_information_mat is not None:
            src2dst_edge.information_mat = new_information_mat
            # dst2src_edge.information_mat = new_information_mat
        if new_rmse is not None:
            src2dst_edge.rmse = new_rmse
            # dst2src_edge.rmse = new_rmse
        write_lock.release()

    def deserialize(self, pose_graph_abstract: Tuple[List[ScanPack], List[PoseGraph_Edge]], adjust_other_nodes=True):
        scans, edges = pose_graph_abstract
        scans_token = []

        all_scan_token = set(s.token for s in self.get_all_scans() + scans)

        for scan in scans:
            scans_token.append(scan.token)
            if (self.has_scan(scan.token)):
                self.update_scan_token(scan_token=scan.token, new_SE3_pred=scan.SE3_pred, new_coor_sys=scan.coor_sys)
            else:
                self.add_vertex(scan)
                logger.info(f'update_graph add a scan ({scan}) to graph')

        for edge in edges:
            edge: PoseGraph_Edge
            if (self.has_edge(edge.src_scan_token, edge.dst_scan_token)):
                self.update_edge_token(edge.src_scan_token, edge.dst_scan_token, new_SE3=edge.SE3)
            elif (self.has_scan(edge.src_scan_token) and self.has_scan(edge.dst_scan_token)):
                self.add_edge(edge)
                logger.info(f'update_graph add a edge ({edge}) to graph')
            else:
                pass

        if (adjust_other_nodes):
            other_nodes = list(filter(lambda scan: scan.token not in scans_token, self.get_all_scans()))
            other_node_tokens = set([s.token for s in other_nodes])
            if (len(other_node_tokens) == 0):
                return
            logger.warning(f'update_graph found ({len(other_nodes)}) node needs adjustment: [' + ','.join([f'{s.token}/{s.type}' for s in other_nodes]) + ']')

            base_scan = self.get_scanpack(self.base_scan_token())

            bfs = [base_scan]
            vis = set()
            while (len(bfs) != 0):
                scan = bfs.pop(0)
                if (scan.token in vis):
                    continue
                else:
                    vis.add(scan.token)
                odom_neighbors = self.get_neighbor_tokens(scan.token)
                for n in odom_neighbors:
                    if (self.has_scan(n) == False):
                        continue
                    scan_neighbor = self.get_scanpack(n)
                    bfs.append(scan_neighbor)
                    if (scan_neighbor.token in other_node_tokens and scan_neighbor.coor_sys != base_scan.coor_sys):
                        edge = self.get_edge(scan.token, scan_neighbor.token)
                        new_SE3 = scan.SE3_pred @ edge.SE3
                        self.update_scan_token(scan_neighbor.token, new_SE3_pred=new_SE3, new_coor_sys=scan.coor_sys)
            assert (other_node_tokens.issubset(vis))
            pass
        return

    def serialize(self):
        scans = []
        for s in self.get_all_scans():
            scans.append(s.copy())
        edges = []
        for e in self.get_all_edges():
            edges.append(e.copy())
        return (scans, edges)

    def get_all_scans(self) -> List[ScanPack]:
        ls = list(self.vertex.values())
        return ls

    def get_all_edges(self) -> List[PoseGraph_Edge]:
        return list(self.edge.values())

    def __global_mapping(self, scan_packs: List[ScanPack], full_pcd: bool, blocking=True):
        """Mapping points to global coordinate (use GPU)

        Args:
            scan_packs (List[ScanPack]): List of ScanPacks to be mapped and merged
            full_pcd (bool): Get full_points (True) or key_points (False)

        Returns:
            (torch.Tensor, torch.Tensor): mapping tile (fea+xyz / xyz+norm+..., N) float and scan_tokens (N, ) long
        """
        map_tile, map_tile_token = [], []
        read_lock = self.locker.gen_wlock()
        read_lock.acquire(blocking=blocking)
        for scan in scan_packs:
            r_scan, t_scan = PoseTool.Rt(scan.SE3_pred.to(self.device))
            if (full_pcd == False):
                if (self.__global_map_cache[scan.token][0] is not None):
                    points = self.__global_map_cache[scan.token][0]
                else:
                    points = scan.key_points.clone().to(self.device)  # clone necessary!
                    points[-3:, :] = r_scan @ points[-3:, :] + t_scan
                    self.__global_map_cache[scan.token][0] = points
            elif (scan.full_pcd is not None):
                if (self.__global_map_cache[scan.token][1] is not None):
                    points = self.__global_map_cache[scan.token][1]
                else:
                    points = scan.full_pcd.clone().to(self.device)  # clone necessary!
                    points[:3, :] = r_scan @ points[:3, :] + t_scan
                    self.__global_map_cache[scan.token][1] = points
            else:
                continue
            map_tile.append(points)
            map_tile_token.append(torch.ones(size=(points.shape[1], ), dtype=torch.long) * scan.token)
        read_lock.release()
        map_tile = torch.concat(map_tile, dim=1).cpu() if len(map_tile) > 0 else None  # (fea+xyz, N)
        map_tile_token = torch.concat(map_tile_token, dim=0).cpu() if len(map_tile_token) > 0 else None  # (N, ), long
        return map_tile, map_tile_token

    def global_map_query_space(self, SE3: torch.Tensor, coor_sys: int, radius: float('inf'), full_pcd: bool = False):
        '''
        SE3: Pose of search point
        radius: Search radius, in meter. infinite search if not given

        return: `[Global_Points, Tokens]` or `[None, None]`
        '''
        if (len(self.vertex) == 0):
            return None, None

        R, t = PoseTool.Rt(SE3)  # (3, 3), (3, 1)

        global_map_scan_list: list[ScanPack] = list(self.vertex.values())  # [scan1, scan2, ...]
        trans_of_scans = torch.stack([PoseTool.Rt(s.SE3_pred)[1] for s in global_map_scan_list], dim=0)  # (N, 3, 1)
        mask = (torch.norm(trans_of_scans - t, dim=1, p=2) < radius).squeeze(dim=1)  # (N, )
        global_map_scan_list = [scan for i, scan in enumerate(global_map_scan_list) if mask[i] == True]
        global_map_scan_list = filter(lambda s: s.coor_sys == coor_sys, global_map_scan_list)

        map_tile, map_tile_token = self.__global_mapping(global_map_scan_list, full_pcd=full_pcd)
        if (map_tile is None):
            return None, None
        assert (map_tile.shape[1] == map_tile_token.shape[0])

        if (full_pcd == False):
            indexs = torch.norm(map_tile[-3:, :] - t, p=2, dim=0) < radius  # (N, )
        else:
            indexs = torch.norm(map_tile[:3, :] - t, p=2, dim=0) < radius  # (N, )
        map_tile = map_tile[:, indexs]  # (fea+xyz, N)
        map_tile_token = map_tile_token[indexs]
        if (full_pcd == False):  # centering
            map_tile[-3:, :] = R.T @ (map_tile[-3:, :] - t)
        else:
            map_tile[:3, :] = R.T @ (map_tile[:3, :] - t)
        assert (map_tile.shape[1] == map_tile_token.shape[0])
        # maptile = R_se3.T @ (R_scan @ p + t_scan - t_se3)
        return map_tile, map_tile_token  # (fea + xyz, N), # (N, )

    def global_map_query_time(self, time: float, radius: float, coor_sys: int, full_pcd: bool = False, centering_SE3: torch.Tensor = torch.eye(4)):
        if (len(self.vertex) == 0):
            return None, None

        # Time Query
        global_map_scan_list: list[ScanPack] = list(self.vertex.values())  # [scan1, scan2, ...]
        global_map_scan_list = filter(lambda scan: abs(scan.timestamp - time) < radius, global_map_scan_list)
        global_map_scan_list = filter(lambda scan: scan.coor_sys == coor_sys, global_map_scan_list)

        # Gathering
        map_tile, map_tile_token = self.__global_mapping(global_map_scan_list, full_pcd=full_pcd)
        if (map_tile is None):
            return None, None

        # Centering
        R, t = PoseTool.Rt(centering_SE3)  # (3, 3), (3, 1)
        if (full_pcd == False):
            map_tile[-3:, :] = R.T @ (map_tile[-3:, :] - t)
        else:
            map_tile[:3, :] = R.T @ (map_tile[:3, :] - t)
        assert (map_tile.shape[1] == map_tile_token.shape[0])
        return map_tile, map_tile_token  # (fea + xyz, N), # (N, )

    def global_map_query_graph(self, token: int, neighbor_level: int, coor_sys: int, max_dist: Union[float, None] = 20, full_pcd: bool = False, centering_SE3: torch.Tensor = torch.eye(4)):
        """Query global map using `Graph` method

        non-keyframe will NOT included.

        Args:
            token (int): token of start scan
            neighbor_level (int): Search level of KNN query
            coor_sys (int): coordinate system
            max_dist (float): max search distance, None for infinity
            full_pcd (bool, optional): Query full-points or key-points only. Defaults to False.
            centering_SE3 (torch.Tensor, optional): SE3 for centering. Defaults to torch.eye(4).

        Returns:
            (Tensor, Tensor):
            Global map (fea + xyz, N),
            Scan Token of points within(N, )
        """
        if (len(self.vertex) == 0):
            return None, None

        center_scan = self.get_scanpack(token)
        center_r, center_t = PoseTool.Rt(center_scan.SE3_pred)

        global_map_scan_list = filter(lambda s: (s.type != 'non-keyframe'), self.graph_search(token=token, neighbor_level=neighbor_level, coor_sys=coor_sys, edge_type=['odom', 'loop']))
        if (max_dist is not None):
            global_map_scan_list = filter(lambda s: (torch.norm(s.SE3_pred[:3, 3:] - center_t, p=2, dim=0).item() < max_dist), global_map_scan_list)

        # Gathering
        map_tile, map_tile_token = self.__global_mapping(global_map_scan_list, full_pcd=full_pcd)
        if (map_tile is None):
            return None, None

        # Centering
        R, t = PoseTool.Rt(centering_SE3)  # (3, 3), (3, 1)
        if (full_pcd == False):
            map_tile[-3:, :] = R.T @ (map_tile[-3:, :] - t)
        else:
            map_tile[:3, :] = R.T @ (map_tile[:3, :] - t)
        assert (map_tile.shape[1] == map_tile_token.shape[0])
        return map_tile, map_tile_token  # (fea + xyz, N), # (N, )

    def graph_search(self, token: int, neighbor_level: int, coor_sys: int, edge_type: Union[Literal['all'], List[str]] = 'all', max_k: Union[None, int] = 16):
        """Search K-Nearest Neighbor in the pose graph

        Args:
            token (int): token of start scan
            neighbor_level (int): Search level of KNN query
            coor_sys (int): coordinate system
            edge_type (str or List[str]): used edge_type
            max_k (int or None): maximum number of frames in knn search. `infinity_length` for unlimited

        Returns:
            List[ScanPack]: Scan pack within KNN region
        """
        if (edge_type == 'all'):
            edge_type = ['loop', 'odom', 'locz', 'prxy']
        global_map_scan_list: Dict[int, ScanPack] = dict()
        bfs = [(neighbor_level, self.get_scanpack(token))]
        while (len(bfs) > 0 and (max_k == None or len(global_map_scan_list) < max_k)):
            level_remain, scanpack = bfs.pop(0) 
            # assert scanpack.coor_sys == coor_sys
            if (scanpack.token in global_map_scan_list.keys()): 
                continue
            global_map_scan_list[scanpack.token] = scanpack 
            if (level_remain <= 0):
                continue
            bfs += [(level_remain - 1, self.get_scanpack(t)) for t in self.get_neighbor_tokens(scanpack.token)
                    if ((self.has_edge(scanpack.token, t) and self.get_edge(scanpack.token, t).type in edge_type) or (
                        self.has_edge(t, scanpack.token) and self.get_edge(t, scanpack.token).type in edge_type))]
        scan_list = list(global_map_scan_list.values())
        return scan_list

    def shortest_path_length(self, src_scan_token, dst_scan_token, edge_type: Union[Literal['all'], List[str]] = 'all', infinity_length: int = 50):
        if src_scan_token == dst_scan_token:
            return 0
        if (edge_type == 'all'):
            edge_type = ['loop', 'odom', 'locz', 'prxy']
        vis: Set[int] = set()
        bfs = [(0, self.get_scanpack(src_scan_token))]
        while (len(bfs) > 0):
            distance, scanpack = bfs.pop(0) 
            if (scanpack.token == dst_scan_token):
                return distance
            if (scanpack.token in vis):  
                continue
            vis.add(scanpack.token) 
            if (distance >= infinity_length):
                continue
            bfs += [(distance + 1, self.get_scanpack(t)) for t in self.get_neighbor_tokens(scanpack.token)
                    if ((self.has_edge(scanpack.token, t) and self.get_edge(scanpack.token, t).type in edge_type) or (
                        self.has_edge(t, scanpack.token) and self.get_edge(t, scanpack.token).type in edge_type))]
        return infinity_length

    def __optim_open3d(self, blocking=True):
        import open3d as o3d
        idx_to_token = dict()
        token_to_idx = dict()
        idx = 0

        base_scan = min(self.vertex.values(), key=lambda s: s.token)

        graph = o3d.pipelines.registration.PoseGraph()

        read_lock = self.locker.gen_rlock()
        read_lock.acquire(blocking=blocking)

        # Add key-frame Scans
        scans = filter(lambda s: s.type != 'non-keyframe', self.get_all_scans())
        for scan in scans:
            v_o3d = o3d.pipelines.registration.PoseGraphNode(pose=scan.SE3_pred)
            graph.nodes.append(v_o3d)
            idx_to_token[idx] = scan.token
            token_to_idx[scan.token] = idx
            idx += 1

        # Add odom/loop/prxy Edges
        edges = filter(lambda e: e.type != 'locz', self.get_all_edges())
        for edge in edges:
            if (edge.src_scan_token in token_to_idx.keys() and edge.dst_scan_token in token_to_idx.keys()):
                e_o3d = o3d.pipelines.registration.PoseGraphEdge(
                    token_to_idx[edge.src_scan_token],
                    token_to_idx[edge.dst_scan_token],
                    np.linalg.inv(edge.SE3),
                    edge.information_mat,
                    # uncertain=False if self.uncertain == False else (False if edge.type == 'odom' else True),
                    uncertain=False,
                )
                graph.edges.append(e_o3d)
            else:
                logger.warning(f'optim find an hanging edge ({edge.src_scan_token}, {edge.dst_scan_token})')

        read_lock.release()

        option = o3d.pipelines.registration.GlobalOptimizationOption(edge_prune_threshold=0.0, preference_loop_closure=2.0, reference_node=token_to_idx[base_scan.token])

        o3d.pipelines.registration.global_optimization(graph, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(), o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                                                       option)

        # fetch result to graph
        diff_t = []
        for idx, v_o3d in enumerate(graph.nodes):
            token = idx_to_token[idx]
            scan = self.get_scanpack(token)

            new_SE3 = torch.as_tensor(graph.nodes[idx].pose.copy()).float()
            # Calculate diff
            diff = torch.norm(scan.SE3_pred[:3, 3:] - new_SE3[:3, 3:], dim=0, p=2)
            scan.SE3_pred = new_SE3
            diff_t.append(diff)

            self.__global_map_cache[token] = [None, None]
            # logger.info(f'Cache of scan {token} is freshed, diff = {diff:.4e}')

        for e_o3d in graph.edges:
            src = idx_to_token[e_o3d.source_node_id]
            dst = idx_to_token[e_o3d.target_node_id]
            T = torch.tensor(e_o3d.transformation, dtype=torch.float)
            self.update_edge_token(src, dst, new_SE3=torch.linalg.inv(T))
            # self.update_edge_token(dst, src, new_SE3=T)

        # write_lock = self.locker.gen_rlock()
        # write_lock.acquire(blocking=blocking)
        # Adjust non-keyframes
        bfs = [base_scan]
        vis: Set[int] = set()
        tokens_tobe_adjusted = set([s.token for s in self.get_all_scans() if s.token not in token_to_idx.keys()])
        while len(bfs) > 0:
            scan = bfs.pop(0)
            if (scan.token in vis):
                continue

            vis.add(scan.token)
            neighbor_token = self.get_neighbor_tokens(scan.token)
            for n in neighbor_token:
                if (self.has_scan(n) == False):
                    continue
                scan_n = self.get_scanpack(n)
                if (scan_n.token in tokens_tobe_adjusted):
                    # assert scan_n.type != 'full', f'optim find a fullscan {scan_n.token}'
                    e = self.get_edge(scan.token, scan_n.token)
                    self.update_scan_token(scan_n.token, new_SE3_pred=scan.SE3_pred @ e.SE3)
                    tokens_tobe_adjusted.remove(scan_n.token)
                if (scan_n.token not in vis):
                    bfs.append(scan_n)
        # write_lock.release()
        assert len(tokens_tobe_adjusted) == 0
        return len(graph.nodes), len(graph.edges), float(sum(diff_t) / len(diff_t))

    def __optim_sesync(self, blocking=True):
        import PySESync
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            pg_file = os.path.join(tmp_dir, 'tmp_posegraphoptim.g2o')
            self.to_g2o_file(pg_file)
            measurements, num_poses = PySESync.read_g2o_file(pg_file)
            logger.info(f"Backend optim loaded {len(measurements)} measurements with {num_poses} poses from tmp_file {pg_file}")

            opts = PySESync.SESyncOpts()
            opts.num_threads = 4
            # opts.verbose=True

            opts.r0 = 3
            opts.formulation = PySESync.Formulation.Explicit  # Options are Simplified or Explicit
            opts.initialization = PySESync.Initialization.Random  # Options are Chordal or Random

            # Termination criteria
            opts.rel_func_decrease_tol = 1e-6
            opts.min_eig_num_tol = 1e-3
            opts.max_time = 5

            result = PySESync.SESync(measurements, opts)

        # fetch result to graph
        diff_t = []
        for idx, v_o3d in enumerate(graph.nodes):
            token = idx_to_token[idx]
            scan = self.get_scanpack(token)

            new_SE3 = torch.as_tensor(graph.nodes[idx].pose.copy()).float()
            # Calculate diff
            diff = torch.norm(scan.SE3_pred[:3, 3:] - new_SE3[:3, 3:], dim=0, p=2)
            scan.SE3_pred = new_SE3
            diff_t.append(diff)

            self.__global_map_cache[token] = [None, None]
            # logger.info(f'Cache of scan {token} is freshed, diff = {diff:.4e}')

        for e_o3d in graph.edges:
            src = idx_to_token[e_o3d.source_node_id]
            dst = idx_to_token[e_o3d.target_node_id]
            T = torch.tensor(e_o3d.transformation, dtype=torch.float)
            self.update_edge_token(src, dst, new_SE3=torch.linalg.inv(T))
            # self.update_edge_token(dst, src, new_SE3=T)

        # write_lock = self.locker.gen_rlock()
        # write_lock.acquire(blocking=blocking)
        # Adjust non-keyframes
        bfs = [base_scan]
        vis: Set[int] = set()
        tokens_tobe_adjusted = set([s.token for s in self.get_all_scans() if s.token not in token_to_idx.keys()])
        while len(bfs) > 0:
            scan = bfs.pop(0)
            if (scan.token in vis):
                continue

            vis.add(scan.token)
            neighbor_token = self.get_neighbor_tokens(scan.token)
            for n in neighbor_token:
                if (self.has_scan(n) == False):
                    continue
                scan_n = self.get_scanpack(n)
                if (scan_n.token in tokens_tobe_adjusted):
                    # assert scan_n.type != 'full', f'optim find a fullscan {scan_n.token}'
                    e = self.get_edge(scan.token, scan_n.token)
                    self.update_scan_token(scan_n.token, new_SE3_pred=scan.SE3_pred @ e.SE3)
                    tokens_tobe_adjusted.remove(scan_n.token)
                if (scan_n.token not in vis):
                    bfs.append(scan_n)
        # write_lock.release()
        assert len(tokens_tobe_adjusted) == 0
        return len(graph.nodes), len(graph.edges), float(sum(diff_t) / len(diff_t))

    def condense(self, base_agent: int, filter_func=Callable[[ScanPack], bool]):
        """Condense the posegraph

        Args:
            filter_func (ScanPack -> bool): whether to select such ScanPack to condensed pose graph

        Returns:
            PoseGraph: Condensed posegraph, with all scans with in `scan_token_list` and one more condensed vertex
        """
        # import pickle as pkl
        # with open('global_posegraph.pkl', 'wb+') as f:
        #     pkl.dump(self, f)

        nx_graph = self.to_networkx()
        condense_pose_graph = PoseGraph(args=self.args, agent_id=self.agent_id, device=self.device)

        scan_list: List[ScanPack] = list(filter(lambda s: filter_func(s), self.get_all_scans()))
        scan_token_list = [s.token for s in scan_list]

        # * add vertex
        scan_list_others = list(filter(lambda s: s.agent_id != base_agent, scan_list))
        scan_agent_ids = set([s.agent_id for s in scan_list_others])

        for scan in scan_list:  # add scan within list
            condense_pose_graph.add_vertex(scan.copy())

        for agent_id in scan_agent_ids:  # add base scans
            base_scan_token = self.base_scan_token(agent_id=agent_id)
            condense_pose_graph.add_vertex(self.get_scanpack(base_scan_token))

        # * add edges
        for agent_id in scan_agent_ids:
            base_scan_token = self.base_scan_token(agent_id=agent_id)
            for scan_of_agent in filter(lambda s: s.agent_id == agent_id, scan_list_others):
                token = scan_of_agent.token
                if (token == base_scan_token):
                    continue
                scan = self.get_scanpack(token)
                path = nx.dijkstra_path(nx_graph, source=base_scan_token, target=scan.token)

                edge_transformation = torch.eye(4)
                information_mat = torch.eye(6)
                confidence = 1
                for i in range(len(path) - 1):
                    src, dst = path[i], path[i + 1]
                    e = self.get_edge(src, dst)
                    edge_transformation = edge_transformation @ e.SE3
                    # TODO(xiaze): check how does information matrix accumulate
                    information_mat = information_mat @ e.information_mat
                    confidence = confidence * e.confidence
                edge = PoseGraph_Edge(src_scan_token=base_scan_token, dst_scan_token=token, SE3=edge_transformation, information_mat=information_mat, type='prxy', confidence=confidence)
                condense_pose_graph.add_edge(edge)

        edges = []
        for edge in self.get_all_edges():
            if (edge.src_scan_token in scan_token_list and edge.dst_scan_token in scan_token_list):
                condense_pose_graph.add_edge(edge)

        return condense_pose_graph

    def subgraph(self, filter_func=Callable[[ScanPack], bool]):
        graph = PoseGraph(self.args, agent_id=self.agent_id, device=self.device)

        scans: List[ScanPack] = []
        for scan in filter(lambda s: filter_func(s), self.get_all_scans()):
            scans.append(scan)
        scan_tokens = set([s.token for s in scans])
        for scan in scans:
            graph.add_vertex(scan)

        for edge in filter(lambda e: (e.src_scan_token in scan_tokens and e.dst_scan_token in scan_tokens), self.get_all_edges()):
            graph.add_edge(edge)
        return graph

    def to_networkx(self):

        graph = nx.Graph()
        nodes = [(s.token, s.type, s.coor_sys, s.agent_id, s.timestep) for s in self.get_all_scans()]
        for token, ntype, ncoor, agentid, timestep in nodes:
            graph.add_node(token, ntype=ntype, ncoor=ncoor, agentid=agentid, timestep=timestep)

        edges = [(e.src_scan_token, e.dst_scan_token, e.type) for e in self.get_all_edges()]
        for src, dst, etype in edges:
            graph.add_edge(src, dst, etype=etype)
        return graph

    def to_g2o_file(self, file_name: str):
        """Store the posegraph into .g2o file

        Args:
            file_path (str): path of file tobe saved
        Reference:
            https://ceres-solver.googlesource.com/ceres-solver/+/1.12.0/examples/slam/common/read_g2o.h
        
        """
        with open(file_name, 'w+') as f:
            for s in self.get_all_scans():
                r, t = PoseTool.Rt(s.SE3_pred)
                quat_r = sci_R.from_matrix(r.cpu().numpy()).as_quat()
                f.write(f'VERTEX_SE3:QUAT {s.token} {t[0,0]} {t[1,0]} {t[2,0]} {quat_r[0]} {quat_r[1]} {quat_r[2]} {quat_r[3]} \n')
            for e in self.get_all_edges():
                r, t = PoseTool.Rt(e.SE3)
                info = e.information_mat
                quat_r = sci_R.from_matrix(r.cpu().numpy()).as_quat()
                f.write(f'EDGE_SE3:QUAT {e.src_scan_token} {e.dst_scan_token}' + f' {t[0,0]} {t[1,0]} {t[2,0]} {quat_r[0]} {quat_r[1]} {quat_r[2]} {quat_r[3]}' +
                        f' {info[0,0]} {info[0,1]} {info[0,2]} {info[0,3]} {info[0,4]} {info[0,5]} ' + f' {info[1,1]} {info[1,2]} {info[1,3]} {info[1,4]} {info[1,5]} ' +
                        f' {info[2,2]} {info[2,3]} {info[2,4]} {info[2,5]} ' + f' {info[3,3]} {info[3,4]} {info[3,5]} ' + f' {info[4,4]} {info[4,5]} ' + f' {info[5,5]} ' + '\n')
        return None

    def repair_coor_sys(self):
        """Assign the correct coor_sys attr for all ScanPack within (INPLACE)
        
        The connected ScanPack pair shall have the same coor_sys attr, thus the repair procedure should apply for
        every multi-agent traj cross.
        """
        not_vis = [s.token for s in self.get_all_scans()]
        while (len(not_vis) > 0):
            seed = min([self.get_scanpack(t) for t in not_vis], key=lambda s: s.coor_sys)
            bfs = [seed.token]
            coor = seed.coor_sys
            while (len(bfs) > 0):
                s = bfs.pop()
                not_vis.remove(s)
                s = self.get_scanpack(s)
                for n in self.get_neighbor_tokens(s.token):
                    if ((n in not_vis) and (n not in bfs)):
                        bfs.append(n)
                if (s.coor_sys != coor):
                    self.update_scan_token(s.token, new_coor_sys=coor)
        return

    def __str__(self):
        scans = self.get_all_scans()
        edges = self.get_all_edges()
        s = f'PoseGraph with {len(scans)} scans and {len(edges)} edges, system_id = {self.agent_id}, base_token = {self.base_scan_token()}'
        return s
