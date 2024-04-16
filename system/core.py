from collections import defaultdict
from enum import Enum, unique
from multiprocessing import Queue
import os
from time import sleep
from typing import Optional, Set
import colorlog as logging
import numpy as np
from tqdm import tqdm

from system.modules.loop_closure import Union, torch
from system.modules.mapping import Union, torch
from system.modules.odometry import Union, torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from easydict import EasyDict
import threading

from scipy.spatial.transform import Rotation as sci_R

from system.modules.odometry import *
from system.modules.mapping import *
from system.modules.loop_closure import *
from system.modules.recoder import ResultLogger
from system.modules.utils import EXIT_CODE, Communicate_Module
from utils.device import detach_to_device


class SlamSystem(object):
    EXTRACTOR_BATCHSIZE = 32
    MAX_CAP_QUEUE = 50

    def __init__(self,
                 args,
                 dpm_encoder: torch.nn.Module,
                 dpm_decoder: torch.nn.Module,
                 system_id: int,
                 logger_dir: Optional[str] = None,
                 comm_module: Optional[Communicate_Module] = None,
                 device: Optional[str] = None) -> None:
        self.args = args
        self.system_id = system_id
        self.coor_sys = self.system_id  # coordinate system index, used in multi-agent SLAM
        self.system_info = EasyDict({'agent_id': self.system_id})

        self.dpm_encoder = dpm_encoder
        self.dpm_decoder = dpm_decoder

        self.frame_id = -1
        self.device = self.args.device if device is None else device
        self.posegraph_map = PoseGraph(args=self.args, agent_id=system_id, device=self.device)

        self.extraction_thread = ExtractionThread(args=self.args.slam_system, posegraph_map=self.posegraph_map, dpm_encoder=self.dpm_encoder, system_info=self.system_info, device=self.device)
        self.odometry_thread = OdometryThread(args=self.args.slam_system, posegraph_map=self.posegraph_map, dpm_decoder=self.dpm_decoder, system_info=self.system_info, device=self.device)
        self.mapping_thread = MappingThread(args=self.args.slam_system, posegraph_map=self.posegraph_map, dpm_decoder=self.dpm_decoder, system_info=self.system_info, device=self.device)
        self.loop_thread = LoopThread(args=self.args.slam_system, posegraph_map=self.posegraph_map, dpm_decoder=self.dpm_decoder, system_info=self.system_info, device=self.device)
        if (logger_dir is None):
            logger_dir = self.args.infer_tgt
        self.result_logger = ResultLogger(args=self.args.slam_system, posegraph_map=self.posegraph_map, log_dir=logger_dir, system_info=self.system_info)

        self.comm_module = comm_module
        if (self.comm_module is not None):
            self.comm_id = self.system_id
            self.comm_module.add_member(self.comm_id)
            logger.info(f'Agent {self.system_id} connected, communicate_id = {self.comm_id}')

        if (self.args.use_ros):
            # Danger zone, system without ROS will raise ModuleNotExistError
            import rospy
            from geometry_msgs.msg import Point, Pose, Quaternion, Twist
            from nav_msgs.msg import Odometry
            from sensor_msgs import point_cloud2
            from sensor_msgs.msg import PointCloud2
            from sensor_msgs.msg import PointField
            from std_msgs.msg import Header

            self.odom_publisher = rospy.Publisher('DeepPointMap_Odometer', data_class=Odometry)
            self.map_publisher = rospy.Publisher('DeepPointMap_GlobalMap', data_class=PointCloud2, queue_size=100)
            # Danger zone, system without ROS will raise ModuleNotExistError

    def MT_Init(self):  # MT = Multi-Threading
        self.MT_queue_input_preprocessing = Queue()
        self.MT_queue_preprocessing_extractor = Queue(maxsize=SlamSystem.MAX_CAP_QUEUE)
        self.MT_queue_extractor_odometer = Queue(maxsize=SlamSystem.MAX_CAP_QUEUE)
        self.MT_queue_odometer_mapping = Queue(maxsize=1)
        self.MT_queue_mapping_backend = Queue(maxsize=SlamSystem.MAX_CAP_QUEUE)
        self.MT_queue_backend_output = Queue(maxsize=SlamSystem.MAX_CAP_QUEUE)

        self.t1 = threading.Thread(target=self.MT_ToDeviceThread, args=[self.MT_queue_input_preprocessing, self.MT_queue_preprocessing_extractor])
        self.t1.start()

        self.t2 = threading.Thread(target=self.MT_ExtractorThread, args=[self.MT_queue_preprocessing_extractor, self.MT_queue_extractor_odometer])
        self.t2.start()

        self.t3 = threading.Thread(target=self.MT_OdometerThread, args=[self.MT_queue_extractor_odometer, self.MT_queue_odometer_mapping])
        self.t3.start()

        self.t4 = threading.Thread(target=self.MT_MappingThread, args=[self.MT_queue_odometer_mapping, self.MT_queue_mapping_backend])
        self.t4.start()

        self.t5 = threading.Thread(target=self.MT_BackendThread, args=[self.MT_queue_mapping_backend, self.MT_queue_backend_output])
        self.t5.start()

        if (self.args.use_ros):
            self.t6 = threading.Thread(target=self.MT_RvizPublisher, args=[self.MT_queue_backend_output])
        else:
            self.t6 = threading.Thread(target=self.MT_PrintThread, args=[self.MT_queue_backend_output])
        self.t6.start()

    def MT_ToDeviceThread(self, queue_in: Queue, queue_out: Queue):
        pbar = tqdm(desc=f'{"ToDevice":<12s}', leave=True, dynamic_ncols=True)
        while (True):
            # Get sensor data
            pbar.total = queue_in.qsize() + pbar.n + 1
            item = queue_in.get()
            if (isinstance(item, EXIT_CODE)):
                queue_out.put(item)
                if (item == EXIT_CODE.exit):
                    break
                else:
                    continue
            else:
                data = item
            pbar.update()

            point_cloud, R, T, padding_mask, original_scan = detach_to_device(data, device='cpu')
            perf_t = time.perf_counter()
            time_ms = len(self.posegraph_map.get_all_scans()) / 10
            self.result_logger.record_perf('to_device', time.perf_counter() - perf_t)
            pbar.set_postfix_str(f'{1000*self.result_logger.get_time_list("to_device")[-1]:2.3f}ms')
            queue_out.put([time_ms, point_cloud, R, T, padding_mask, original_scan])
        pbar.close()

    def MT_ExtractorThread(self, queue_in: Queue, queue_out: Queue):
        pbar = tqdm(desc=f'{"Extract":<12s}', leave=True, dynamic_ncols=True)
        will_exit = False
        while (will_exit == False):
            # DPM Encoder (extract descriptors)

            input_pcds = [queue_in.get()]
            while (queue_in.qsize() > 0 and len(input_pcds) < SlamSystem.EXTRACTOR_BATCHSIZE):
                input_pcds.append(queue_in.get())
            pbar.set_description(f'{"Extract"+f" ({len(input_pcds)})":<12s}')
            point_clouds, padding_masks, Rs, Ts, times = [], [], [], [], []

            for item in input_pcds:
                if (isinstance(item, EXIT_CODE)):
                    queue_out.put(item)
                    if (item == EXIT_CODE.exit):
                        will_exit = True
                    else:
                        continue
                else:
                    time_ms, point_cloud, R, T, padding_mask, original_scan = item
                    point_clouds.append(point_cloud)
                    padding_masks.append(padding_mask)
                    Rs.append(R)
                    Ts.append(T)
                    times.append(time_ms)

                    pbar.total = queue_in.qsize() + pbar.n + 1
                    pbar.update()
            perf_t = time.perf_counter()
            if (len(point_clouds) > 0):  # for last exit_code
                point_clouds = torch.concat(point_clouds, dim=0)  # B, 3, N
                padding_masks = torch.concat(padding_masks, dim=0)  # B, N

                descriptors_batch = self.extraction_thread.process(point_cloud=point_clouds, padding_mask=padding_masks)  # B, 128+3, N

                for time_ms, descriptors, point_cloud, R, T in zip(times, descriptors_batch, point_clouds, Rs, Ts):
                    self.frame_id += 1
                    new_scan = ScanPack(
                        timestamp=self.frame_id * 0.1,
                        agent_id=self.system_id,
                        # timestep=self.posegraph_map.all_frame_num,
                        timestep=self.frame_id,
                        key_points=descriptors,  # fea+xyz, N
                        full_pcd=point_cloud.clone() * self.args.slam_system.coor_scale,  # N, 3
                        coor_sys=self.coor_sys,
                        SE3_gt=PoseTool.SE3(R, T))

                    queue_out.put(new_scan.to('cpu'))
            self.result_logger.record_perf('extract', time.perf_counter() - perf_t)
            pbar.set_postfix_str(f'{1000*self.result_logger.get_time_list("extract")[-1]:2.3f}ms')
        pbar.close()

    def MT_OdometerThread(self, queue_in: Queue, queue_out: Queue):
        pbar = tqdm(desc=f'{"Odometry":<12s}', leave=True, dynamic_ncols=True)
        while (True):
            # Odometer
            pbar.total = queue_in.qsize() + pbar.n + 1
            item = queue_in.get()
            if (isinstance(item, EXIT_CODE)):
                queue_out.put(item)
                if (item == EXIT_CODE.exit):
                    break
                else:
                    continue
            else:
                new_scan = item
            pbar.update()

            perf_t = time.perf_counter()
            odom_edges = self.odometry_thread.process(new_scan=new_scan)
            if len(odom_edges) == 0:
                new_scan.SE3_pred = torch.eye(4)
                self.posegraph_map.add_vertex(new_scan)
                self.posegraph_map.last_known_anyframe = new_scan.token
                self.posegraph_map.last_known_keyframe = new_scan.token
                queue_out.put(EXIT_CODE.acpt)
                continue
            else:
                odom_edge = odom_edges[0]  # Assert odometry edge contains only one edge
            self.result_logger.record_perf('odometer', time.perf_counter() - perf_t)
            pbar.set_postfix_str(f'{1000*self.result_logger.get_time_list("odometer")[-1]:2.3f}ms')
            queue_out.put([new_scan.to('cpu'), odom_edge.to('cpu')])
        pbar.close()

    def MT_MappingThread(self, queue_in: Queue, queue_out: Queue):
        pbar = tqdm(desc=f'{"Mapping":<12s}', leave=True, dynamic_ncols=True)
        while (True):
            # Mapping (drop + keyframe + s2m)
            pbar.total = queue_in.qsize() + pbar.n + 1
            item = queue_in.get()
            if (isinstance(item, EXIT_CODE)):
                queue_out.put(item)
                if (item == EXIT_CODE.exit):
                    break
                else:
                    continue
            else:
                new_scan: ScanPack = item[0]
                odom_edge: PoseGraph_Edge = item[1]

            perf_t = time.perf_counter()
            result = self.mapping_thread.process(new_scan=new_scan, odom_edge=odom_edge)
            self.result_logger.record_perf('mapping', time.perf_counter() - perf_t)
            pbar.set_postfix_str(f'{1000*self.result_logger.get_time_list("mapping")[-1]:2.3f}ms')

            if (isinstance(result, EXIT_CODE)):
                queue_out.put(result)
            else:
                queue_out.put(new_scan.to('cpu'))
                pbar.update()
        pbar.close()

    def MT_BackendThread(self, queue_in: Queue, queue_out: Queue):
        pbar = tqdm(desc=f'{"Backend":<12s}', leave=True, dynamic_ncols=True)
        while (True):
            # Loop-Closure and optim
            pbar.total = queue_in.qsize() + pbar.n + 1
            item = queue_in.get()
            if (isinstance(item, EXIT_CODE)):
                queue_out.put(item)
                if (item == EXIT_CODE.exit):
                    break
                else:
                    continue
            else:
                new_scan = item
            pbar.update()

            perf_t = time.perf_counter()
            self.loop_thread.process(new_scan=new_scan, targets='all')
            self.posegraph_map.last_known_anyframe = new_scan.token
            self.result_logger.record_perf('loop_closure', time.perf_counter() - perf_t)
            pbar.set_postfix_str(f'{1000*self.result_logger.get_time_list("loop_closure")[-1]:2.3f}ms')
            queue_out.put(EXIT_CODE.acpt)
        pbar.close()

    def MT_PrintThread(self, queue_in: Queue):
        pbar = tqdm(desc=f'{"Output":<12s}', leave=True, dynamic_ncols=True)
        while (True):
            pbar.total = queue_in.qsize() + pbar.n + 1
            pbar.update()
            item = queue_in.get()
            pbar.set_description_str(f'{"Output":<12s}' + f'{str(item):<12s}')
            if (item == EXIT_CODE.exit):
                break
        pbar.close()

    def MT_RvizPublisher(self, queue_in: Queue):
        import rospy
        from geometry_msgs.msg import Point, Pose, Quaternion, Twist
        from nav_msgs.msg import Odometry
        from sensor_msgs import point_cloud2
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Header

        assert self.args.use_ros
        pbar = tqdm(desc=f'{"Rviz":<12s}', leave=True, dynamic_ncols=True)
        while (True):
            pbar.total = queue_in.qsize() + pbar.n + 1
            pbar.update()
            item = queue_in.get()
            pbar.set_description_str(f'{"Rviz":<12s}' + f'{str(item):<12s}')
            if (item == EXIT_CODE.exit):
                break
            else:
                # output to RVIZ

                current_scan = self.posegraph_map.get_scanpack(scan_token=self.posegraph_map.last_known_anyframe)
                assert (current_scan.SE3_pred is not None)
                r, t = PoseTool.Rt(current_scan.SE3_pred)
                q = sci_R.from_matrix(r.cpu().numpy()).as_quat()  # xyzw

                msg = Odometry()
                msg.header.frame_id = 'map'
                msg.pose.pose = Pose(Point(x=t[0, 0], y=t[1, 0], z=t[2, 0]), Quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
                self.odom_publisher.publish(msg)

                if (self.posegraph_map.all_frame_num % 1 == 0):
                    last_known_keyframe = self.posegraph_map.get_scanpack(self.posegraph_map.last_known_keyframe)
                    src_map_descriptors, src_map_src_map_descriptor_token = self.posegraph_map.global_map_query_graph(token=last_known_keyframe.token,
                                                                                                                      neighbor_level=5,
                                                                                                                      coor_sys=last_known_keyframe.coor_sys,
                                                                                                                      max_dist=50,
                                                                                                                      full_pcd=True,
                                                                                                                      centering_SE3=current_scan.SE3_pred)

                    if (src_map_descriptors is not None):
                        fields = [
                            PointField("x", 0, PointField.FLOAT32, 1),
                            PointField("y", 4, PointField.FLOAT32, 1),
                            PointField("z", 8, PointField.FLOAT32, 1),
                        ]
                        header = Header()
                        header.frame_id = "rslidar"
                        header.stamp = rospy.Time.now()
                        points = torch.concat([src_map_descriptors[:3, :], current_scan.full_pcd[:3, :]], dim=-1).T.numpy()  # N, 3
                        pc2 = point_cloud2.create_cloud(header, fields, points)
                        self.map_publisher.publish(pc2)

        pbar.close()

    def MT_Step(self, sensor_data: List[torch.Tensor]):
        self.MT_queue_input_preprocessing.put(sensor_data)

    def MT_Done(self):
        self.MT_queue_input_preprocessing.put(EXIT_CODE.exit)

    def MT_Wait(self):
        self.t1.join()
        self.t2.join()
        self.t3.join()
        self.t4.join()
        self.t5.join()
        self.t6.join()

        self.MT_queue_input_preprocessing.close()
        self.MT_queue_preprocessing_extractor.close()
        self.MT_queue_extractor_odometer.close()
        self.MT_queue_odometer_mapping.close()
        self.MT_queue_mapping_backend.close()
        self.MT_queue_backend_output.close()

        logger.critical(f"All thread finished.")

    def step(self, sensor_data: List[torch.Tensor]) -> EXIT_CODE:
        self.frame_id += 1
        # Get sensor data
        perf_t = time.perf_counter()
        time_ms = len(self.posegraph_map.get_all_scans()) / 10
        point_cloud, R, T, padding_mask, original_scan = detach_to_device(sensor_data, device=self.device, non_blocking=True)
        self.result_logger.record_perf('to_device', time.perf_counter() - perf_t)

        # DPM Encoder (extract descriptors)
        perf_t = time.perf_counter()
        descriptors = self.extraction_thread.process(point_cloud=point_cloud, padding_mask=padding_mask)
        new_scan = ScanPack(
            timestamp=time_ms,
            agent_id=self.system_id,
            # timestep=self.posegraph_map.all_frame_num,
            timestep=self.frame_id,
            key_points=descriptors[0],  # [fea+xyz]
            full_pcd=point_cloud[0].clone() * self.args.slam_system.coor_scale,  # N, 3
            coor_sys=self.coor_sys,
            SE3_gt=PoseTool.SE3(R[0], T[0]))
        self.result_logger.record_perf('extract', time.perf_counter() - perf_t)

        # Odometer
        perf_t = time.perf_counter()
        odom_edges = self.odometry_thread.process(new_scan=new_scan)
        if len(odom_edges) == 0:
            new_scan.SE3_pred = torch.eye(4)
            self.posegraph_map.add_vertex(new_scan)
            self.posegraph_map.last_known_anyframe = new_scan.token
            self.posegraph_map.last_known_keyframe = new_scan.token
            odom_edge = None
        else:
            odom_edge = odom_edges[0]  # Assert odometry edge contains only one edge
            self.result_logger.record_perf('odometer', time.perf_counter() - perf_t)

            # Mapping (drop + keyframe + s2m)
            # Try: Dynamic Frame merge
            perf_t = time.perf_counter()
            result = self.mapping_thread.process(new_scan=new_scan, odom_edge=odom_edge)
            if isinstance(result, EXIT_CODE):
                return result
            else:
                assert isinstance(result, PoseGraph_Edge)
            self.result_logger.record_perf('mapping', time.perf_counter() - perf_t)

            # Loop-Closure and optim
            perf_t = time.perf_counter()
            self.loop_thread.process(new_scan=new_scan, targets='self')
            self.posegraph_map.last_known_anyframe = new_scan.token
            self.result_logger.record_perf('loop_closure', time.perf_counter() - perf_t)

        if (self.comm_module is not None):
            neibor_edges = []  # without odom edge
            for j in self.posegraph_map.get_neighbor_tokens(new_scan.token):
                if (j == odom_edge.src_scan_token or j == odom_edge.dst_scan_token):
                    continue
                if (self.posegraph_map.has_edge(j, new_scan.token)):
                    neibor_edges.append(self.posegraph_map.get_edge(j, new_scan.token))
                elif (self.posegraph_map.has_edge(new_scan.token, j)):
                    neibor_edges.append(self.posegraph_map.get_edge(new_scan.token, j))
                else:
                    raise RuntimeError(f'both edge {(new_scan.token,j)} and {(j, new_scan.token)} not exists')
            self.comm_module.send_message(caller=self.comm_id, callee=0, command='UPLOAD_SCAN', message=dict(new_scan=new_scan, odometer_edge=odom_edge, neighbor_edges=neibor_edges))
        return EXIT_CODE.acpt


class AgentSystem(SlamSystem):
    def __init__(self,
                 args,
                 dpm_encoder: torch.nn.Module,
                 dpm_decoder: torch.nn.Module,
                 system_id: int,
                 logger_dir: Optional[str] = None,
                 comm_module: Optional[Communicate_Module] = None,
                 device: Optional[str] = None) -> None:
        super().__init__(args, dpm_encoder, dpm_decoder, system_id, logger_dir, comm_module, device)

    def start(self, dataloader):
        def feed_data_loop():
            infer_loop = tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, desc=f'Agent {self.system_id}')
            for frame_id, data in enumerate(infer_loop):
                code = self.step(data)
                infer_loop.set_description_str(f"Agent {self.system_id}: [{code}]" + ", ".join([f"{name}:{time[0]:.3f}s" for name, time in self.result_logger.log_time(window=50).items()]))
            infer_loop.set_description_str(f"Agent {self.system_id}: [DONE]" + ", ".join([f"{name}:{time[0]:.3f}s" for name, time in self.result_logger.log_time().items()]))
        self.__thread = threading.Thread(target=feed_data_loop)
        self.__thread.start()

    def wait(self):
        self.__thread.join()


class CloudSystem(SlamSystem):
    def __init__(self,
                 args,
                 dpm_encoder: torch.nn.Module,
                 dpm_decoder: torch.nn.Module,
                 logger_dir: Optional[str] = None,
                 comm_module: Optional[Communicate_Module] = None,
                 device: Optional[str] = None) -> None:
        super().__init__(args=args, dpm_encoder=dpm_encoder, dpm_decoder=dpm_decoder, system_id=0, logger_dir=logger_dir, comm_module=comm_module, device=device)

        assert comm_module is not None
        self.communicate_module = comm_module
        self.posegraph_map.agent_id = 0
        self.posegraph_map.uncertain = True

    def step(self, scan_pack: ScanPack, odom_edge: Optional[PoseGraph_Edge], neighbor_edges: List[PoseGraph_Edge]):
        assert scan_pack.type == 'full'
        
        #* Step1: Add scan and edges into pose graph
        self.posegraph_map.add_vertex(scan_pack)
        if (odom_edge is not None):
            assert scan_pack.token in [odom_edge.src_scan_token, odom_edge.dst_scan_token]
            if (scan_pack.token == odom_edge.src_scan_token):
                assert self.posegraph_map.has_scan(odom_edge.dst_scan_token)
                dst_scan = self.posegraph_map.get_scanpack(odom_edge.dst_scan_token)
                SE3 = dst_scan.SE3_pred @ torch.linalg.inv(odom_edge.SE3)
                self.posegraph_map.update_scan_token(scan_token=scan_pack.token, new_SE3_pred=SE3, new_coor_sys=dst_scan.coor_sys)
            elif (scan_pack.token == odom_edge.dst_scan_token):
                # src_scan_token @ edge = scan_pack
                assert self.posegraph_map.has_scan(odom_edge.src_scan_token)
                src_scan = self.posegraph_map.get_scanpack(odom_edge.src_scan_token)
                SE3 = src_scan.SE3_pred @ odom_edge.SE3
                self.posegraph_map.update_scan_token(scan_token=scan_pack.token, new_SE3_pred=SE3, new_coor_sys=src_scan.coor_sys)
            self.posegraph_map.add_edge(odom_edge)
        for e in neighbor_edges:
            self.posegraph_map.add_edge(e)

        #* Step2: Use Coor transformation to adjust SE3 in scanpack (for those agent which didnt update posegraph in time)
        scan_base = min(filter(lambda s: s.agent_id == scan_pack.agent_id, self.posegraph_map.get_all_scans()), key=lambda s: s.timestep)
        if (scan_base.coor_sys != scan_pack.coor_sys):
            logger.critical(f'system {self.system_id}: '
                           f'Received a scan ({scan_pack.agent_id}-{scan_pack.timestep}@{scan_pack.token}/{scan_pack.coor_sys}) with coor {scan_pack.coor_sys}, '
                           f'which should be {scan_base.coor_sys} (base = {scan_base.agent_id}-{scan_base.timestep}@{scan_base.token}/{scan_base.coor_sys})'
                           f'auto merging...')
            _neighbor = self.posegraph_map.get_neighbor_tokens(scan_pack.token)
            for _n_token in _neighbor:
                _nei_scan = self.posegraph_map.get_scanpack(_n_token)
                _e = self.posegraph_map.get_edge(_n_token, scan_pack.token)
                _pose_new = _nei_scan.SE3_pred @ _e.SE3
                _coor_new = _nei_scan.coor_sys
                _dr, _dt = PoseTool.Rt(_pose_new @ scan_pack.SE3_pred.inverse())
                _dr = PoseTool.rotation_angle(_dr)
                logger.info(f'Cloud received an out-of-date scan ({scan_pack.agent_id}-{scan_pack.timestep}@{scan_pack.token}/{scan_pack.coor_sys}), '
                            f'Adjustment delta = {(_dr*180/torch.pi):.3f}D, {_dt.abs().max():.3f}m')
            self.posegraph_map.update_scan_token(scan_pack.token, new_SE3_pred=_pose_new, new_coor_sys=_coor_new)

        #* Step3: Make multi-agent loop closure
        validated_loop_edges = self.loop_thread.process(scan_pack, targets='others')
        for e in validated_loop_edges:
            if (self.posegraph_map.has_edge(e.src_scan_token, e.dst_scan_token) == False):
                self.posegraph_map.add_edge(e)
        # if (len(validated_loop_edges) > 0):
        #     self.result_logger.draw_trajectory(f'cloud_0_test_{self.posegraph_map.key_frame_num}')
        return

    def start(self):
        def fetch_data_loop():
            tq = tqdm(desc=f'Cloud {self.system_id}', leave=False, dynamic_ncols=True)
            while True:
                (command, data) = self.communicate_module.fetch_message(self.system_id, block=True)
                if(command == 'QUIT'):
                    break
                elif (command == 'NO_OP'):
                    # sleep(0.1)  # 10fps busy-wait
                    continue
                elif (command == 'AGENT_QUIT' and data is None):
                    logger.warning(f'"AGENT_QUIT" received')
                    continue
                elif (command == 'UPLOAD_SCAN'):
                    scan_pack, odom_edge, neighbor_edges = data['new_scan'], data['odometer_edge'], data['neighbor_edges']
                    self.step(scan_pack=scan_pack, odom_edge=odom_edge, neighbor_edges=neighbor_edges)
                    tq.total = self.communicate_module.get_queue_length(self.system_id) + self.posegraph_map.key_frame_num
                    tq.update()
                else:
                    raise RuntimeError(f'unknown operation code {command} to cloud {self.system_id}')
            self.result_logger.save_trajectory('cloud_0_traj')
            self.result_logger.draw_trajectory('cloud_0_traj')
            self.result_logger.save_map('cloud_0_map')

        self.__thread = threading.Thread(target=fetch_data_loop)
        self.__thread.start()

    def wait(self):
        self.__thread.join()
        logger.critical(f'Cloud:{self.system_id} exited.')
        
        