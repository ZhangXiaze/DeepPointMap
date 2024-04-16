import colorlog as logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import random
import pickle
import numpy as np
import numpy.linalg as linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as Tensor
from typing import Tuple, Dict
from utils.pose import rt_global_to_relative


class DeepPointModelPipeline(nn.Module):


    def __init__(self, args, encoder: nn.Module, decoder: nn.Module, criterion: nn.Module):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = criterion
        self._forward_method = None
        self.registration()
        self.refined_SE3_cache = dict()

    def forward(self, *args) -> Tuple[Tensor, dict]:
        return self._forward_method(*args)

    def _train_registration(self, pcd: Tensor, R: Tensor, T: Tensor, padding_mask: Tensor, calib: Tensor, info: dict)\
            -> Tuple[Tensor, dict]:

        coor, fea, mask = self.encoder(pcd, padding_mask)
        S, _, N = coor.shape
        B = info['num_map']
        S = S // B
        coor = coor * self.args.slam_system.coor_scale
        pcd_index = np.asarray([dsf_index[2] for dsf_index in info['dsf_index']])  # len = pcd.shape[0]
        refined_SE3_file = info['refined_SE3_file']  # len = B

        fea = fea.reshape(B, S, -1, N)
        coor = coor.reshape(B, S, -1, N)
        R = R.reshape(B, S, 3, 3)
        T = T.reshape(B, S, 3, 1)
        mask = mask.reshape(B, S, -1)
        pcd_index = pcd_index.reshape((B, S))
        calib = calib.reshape(B, S, 4, 4)

        map_size_max = self.args.train.registration.map_size_max
        if S <= map_size_max:
            if random.random() < 0.5:
                S1 = 1
            else:
                S1 = random.randint(1, S - 1)  
        else:
            S1 = random.randint(S - map_size_max, map_size_max)  
        S2 = S - S1

        src_coor, dst_coor = coor[:, :S1], coor[:, S1:]
        src_fea, dst_fea = fea[:, :S1], fea[:, S1:]
        src_R, dst_R = R[:, :S1], R[:, S1:]
        src_T, dst_T = T[:, :S1], T[:, S1:]
        src_mask, dst_mask = mask[:, :S1], mask[:, S1:]
        src_index, dst_index = pcd_index[:, :S1], pcd_index[:, S1:]
        src_calib, dst_calib = calib[:, :S1], calib[:, S1:]

        R, T = self._get_accurate_RT(src_index=src_index[:, 0], dst_index=dst_index[:, 0],
                                     src_R=src_R[:, 0], src_T=src_T[:, 0], src_calib=src_calib[:, 0],
                                     dst_R=dst_R[:, 0], dst_T=dst_T[:, 0], dst_calib=dst_calib[:, 0],
                                     # src_pcd=src_pcd[:, 0], dst_pcd=dst_pcd[:, 0],
                                     refined_SE3_file=refined_SE3_file)
        if S1 > 1:

            map1_relative_R, map1_relative_T = \
                self._get_accurate_RT(src_index=src_index[:, 1:], dst_index=src_index[:, 0],
                                      src_R=src_R[:, 1:], src_T=src_T[:, 1:], src_calib=src_calib[:, 1:],
                                      dst_R=src_R[:, 0], dst_T=src_T[:, 0], dst_calib=src_calib[:, 0],
                                      # src_pcd=src_pcd[:, 1:], dst_pcd=src_pcd[:, 0],
                                      refined_SE3_file=refined_SE3_file)

            src_coor[:, 1:] = map1_relative_R @ src_coor[:, 1:] + map1_relative_T

        if S2 > 1:

            map2_relative_R, map2_relative_T = \
                self._get_accurate_RT(src_index=dst_index[:, 1:], dst_index=dst_index[:, 0],
                                      src_R=dst_R[:, 1:], src_T=dst_T[:, 1:], src_calib=dst_calib[:, 1:],
                                      dst_R=dst_R[:, 0], dst_T=dst_T[:, 0], dst_calib=dst_calib[:, 0],
                                      refined_SE3_file=refined_SE3_file,
                                      # src_pcd=dst_pcd[:, 1:], dst_pcd=dst_pcd[:, 0],
                                      bridge_index=src_index[:, 0]) 
            dst_coor[:, 1:] = map2_relative_R @ dst_coor[:, 1:] + map2_relative_T

        src_coor = src_coor.transpose(1, 2).reshape(B, -1, S1 * N)
        src_fea = src_fea.transpose(1, 2).reshape(B, -1, S1 * N)
        src_mask = src_mask.reshape(B, -1)
        dst_coor = dst_coor.transpose(1, 2).reshape(B, -1, S2 * N)
        dst_fea = dst_fea.transpose(1, 2).reshape(B, -1, S2 * N)
        dst_mask = dst_mask.reshape(B, -1)
        src_global_coor = R @ src_coor + T
        dst_global_coor = dst_coor.clone()


        src_pairing_fea, dst_pairing_fea, src_coarse_pairing_fea, dst_coarse_pairing_fea, \
        src_offset_res, dst_offset_res = \
            self.decoder(
                torch.cat([src_fea, src_coor], dim=1),
                torch.cat([dst_fea, dst_coor], dim=1),
                src_padding_mask=src_mask,
                dst_padding_mask=dst_mask,
                gt_Rt=(R, T)
            )
        loss, top1_pairing_acc, loss_pairing, loss_coarse_pairing, loss_offset = \
            self.criterion(
                src_global_coor=src_global_coor, dst_global_coor=dst_global_coor,
                src_padding_mask=src_mask, dst_padding_mask=dst_mask,
                src_pairing_fea=src_pairing_fea, dst_pairing_fea=dst_pairing_fea,
                src_coarse_pairing_fea=src_coarse_pairing_fea, dst_coarse_pairing_fea=dst_coarse_pairing_fea,
                src_offset_res=src_offset_res, dst_offset_res=dst_offset_res
            )
        offset_err = (torch.norm(src_offset_res.detach(), p=2, dim=1).mean() +
                      torch.norm(dst_offset_res.detach(), p=2, dim=1).mean()).item() / 2
        metric_dict = {
            'loss_regis': loss.item(),
            'loss_p': loss_pairing.item(),
            'loss_c': loss_coarse_pairing.item(),
            'loss_o': loss_offset.item(),
            'top1_acc': top1_pairing_acc,
            'offset_err': offset_err
        }
        return loss, metric_dict

    def _train_loop_detection(self, src_pcd: Tensor, src_R: Tensor, src_T: Tensor, src_mask: Tensor, src_calib: Tensor,
                              dst_pcd: Tensor, dst_R: Tensor, dst_T: Tensor, dst_mask: Tensor, dst_calib: Tensor
                              ) -> Tuple[Tensor, dict]:
        B = src_pcd.shape[0]
        stacked_pcd = torch.cat([src_pcd, dst_pcd], dim=0)  # (2B, C, N)
        stacked_mask = torch.cat([src_mask, dst_mask], dim=0)  # (2B, N)

        coor, fea, mask = self.encoder(stacked_pcd, stacked_mask)
        coor = coor * self.args.slam_system.coor_scale 
        src_coor, dst_coor = coor[:B], coor[B:]
        src_fea, dst_fea = fea[:B], fea[B:]
        src_mask, dst_mask = mask[:B], mask[B:]

        src_descriptor = torch.cat([src_fea, src_coor], dim=1)
        dst_descriptor = torch.cat([dst_fea, dst_coor], dim=1)
        loop_pred = self.decoder.loop_detection_forward(
            src_descriptor=src_descriptor, dst_descriptor=dst_descriptor,
            src_padding_mask=src_mask, dst_padding_mask=dst_mask,
        )

        dis = torch.norm((src_T - dst_T).squeeze(-1), p=2, dim=-1)
        loop_gt = (dis <= self.args.train.loop_detection.distance).float()
        loop_loss = F.binary_cross_entropy(input=loop_pred, target=loop_gt)

        loop_pred_binary = loop_pred > 0.5
        loop_gt_binary = loop_gt.bool()
        precision = (torch.sum(loop_pred_binary == loop_gt_binary) / loop_pred_binary.shape[0]).item()
        if loop_gt_binary.sum() > 0:
            recall = torch.sum(loop_pred_binary[loop_gt_binary]) / loop_gt_binary.sum()
            recall = recall.item()
        else:
            recall = 1.0
        negative_gt_mask = ~loop_gt_binary
        if negative_gt_mask.sum() > 0:
            false_positive = torch.sum(loop_pred_binary[negative_gt_mask]) / negative_gt_mask.sum()
            false_positive = false_positive.item()
        else:
            false_positive = 0.0

        metric_dict = {
            'loss_loop': loop_loss.item(),
            'loop_precision': precision,
            'loop_recall': recall,
            'loop_false_positive': false_positive
        }
        return loop_loss, metric_dict

    def registration(self):
        self._forward_method = self._train_registration
        for name, param in self.named_parameters():
            if 'loop' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def loop_detection(self):
        self._forward_method = self._train_loop_detection
        for name, param in self.named_parameters():
            if 'loop' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _get_accurate_RT(self, src_index: np.ndarray, dst_index: np.ndarray, refined_SE3_file: str,
                         src_R: Tensor, src_T: Tensor, src_calib: Tensor,
                         dst_R: Tensor, dst_T: Tensor, dst_calib: Tensor, bridge_index=None,
                         src_pcd=None, dst_pcd=None) -> Tuple[Tensor, Tensor]:
        assert len(src_index) == len(dst_index) == len(refined_SE3_file) == len(src_R) == len(src_T) == len(src_calib) \
               == len(dst_R) == len(dst_T) == len(dst_calib)
        B = len(src_index)
        device = src_calib.device
        if bridge_index is None:
            bridge_index = [None] * B
        else:
            assert len(bridge_index) == B
        use_squeeze = src_index.ndim == 1 and dst_index.ndim == 1
        src_index = src_index[:, np.newaxis] if src_index.ndim == 1 else src_index
        dst_index = dst_index[:, np.newaxis] if dst_index.ndim == 1 else dst_index
        src_R = src_R.unsqueeze(1) if src_R.ndim == 3 else src_R
        src_T = src_T.unsqueeze(1) if src_T.ndim == 3 else src_T
        src_calib = src_calib.unsqueeze(1) if src_calib.ndim == 3 else src_calib
        dst_R = dst_R.unsqueeze(1) if dst_R.ndim == 3 else dst_R
        dst_T = dst_T.unsqueeze(1) if dst_T.ndim == 3 else dst_T
        dst_calib = dst_calib.unsqueeze(1) if dst_calib.ndim == 3 else dst_calib

        S = max(src_index.shape[1], dst_index.shape[1])
        if src_index.shape[1] < S and src_index.shape[1] == 1:
            src_index = src_index.repeat(repeats=S, axis=1)
            src_R = src_R.repeat(1, S, 1, 1)
            src_T = src_T.repeat(1, S, 1, 1)
            src_calib = src_calib.repeat(1, S, 1, 1)

        if dst_index.shape[1] < S and dst_index.shape[1] == 1:
            dst_index = dst_index.repeat(repeats=S, axis=1)
            dst_R = dst_R.repeat(1, S, 1, 1)
            dst_T = dst_T.repeat(1, S, 1, 1)
            dst_calib = dst_calib.repeat(1, S, 1, 1)

        R_list, T_list = [], []
        for b_src_i, b_src_R, b_src_T, b_src_c, b_dst_i, b_dst_R, b_dst_T, b_dst_c, file, bridge in \
            zip(src_index, src_R, src_T, src_calib, dst_index, dst_R, dst_T, dst_calib, refined_SE3_file, bridge_index):
            SE3_dict = self._load_refined_SE3(file)
            if SE3_dict is not None:
                b_SE3 = []
                for i, (s, d, s_calib, d_calib) in enumerate(zip(b_src_i, b_dst_i, b_src_c, b_dst_c)):
                    try:
                        icp_SE3 = torch.from_numpy(get_SE3_from_dict(SE3_dict, s, d, bridge)).float().to(device)
                        current_SE3 = d_calib @ icp_SE3 @ s_calib.inverse()
                    except:
                        r, t = rt_global_to_relative(center_R=b_dst_R[i], center_T=b_dst_T[i],
                                                     other_R=b_src_R[i], other_T=b_src_T[i])
                        current_SE3 = torch.eye(4, dtype=torch.float32, device=device)
                        current_SE3[:3, :3] = r
                        current_SE3[:3, 3:] = t
                        import os
                        src_SE3 = torch.eye(4, dtype=torch.float32, device=device)
                        dst_SE3 = torch.eye(4, dtype=torch.float32, device=device)
                        src_SE3[:3, :3] = b_src_R[i]
                        src_SE3[:3, 3:] = b_src_T[i]
                        dst_SE3[:3, :3] = b_dst_R[i]
                        dst_SE3[:3, 3:] = b_dst_T[i]
                        gt_ori_relative_SE3 = d_calib.inverse() @ dst_SE3.inverse() @ src_SE3 @ s_calib
                        dist = torch.norm(gt_ori_relative_SE3[:3, -1]).item()
                    b_SE3.append(current_SE3)
                b_SE3 = torch.stack(b_SE3, dim=0)
                R_list.append(b_SE3[:, :3, :3])
                T_list.append(b_SE3[:, :3, 3:])
            else: 
                R, T = rt_global_to_relative(center_R=b_dst_R, center_T=b_dst_T, other_R=b_src_R, other_T=b_src_T)
                R_list.append(R)
                T_list.append(T)


        R, T = torch.stack(R_list, dim=0), torch.stack(T_list, dim=0)
        if use_squeeze:
            R, T = R.squeeze(1), T.squeeze(1)
        return R, T

    def _load_refined_SE3(self, file):
        if file not in self.refined_SE3_cache.keys():
            if file != '':
                with open(file, 'rb') as f:
                    refined_SE3: Dict[Tuple[int, int], np.ndarray] = pickle.load(f)
            else:
                refined_SE3 = None
            self.refined_SE3_cache[file] = refined_SE3
        return self.refined_SE3_cache[file]


def get_SE3_from_dict(SE3_dict: Dict[Tuple[int, int], np.ndarray], s: int, d: int, bridge=None) -> np.ndarray:
    if s == d:
        SE3 = np.eye(4)
    elif s < d:
        SE3 = SE3_dict.get((s, d), None)
        if SE3 is not None:
            SE3 = linalg.inv(SE3)
    else:
        SE3 = SE3_dict.get((d, s), None)
    if SE3 is None:
        SE3_s2b = get_SE3_from_dict(SE3_dict, s, bridge, None)  # get s -> bridge
        SE3_b2d = get_SE3_from_dict(SE3_dict, bridge, d, None)  # get bridge -> d
        SE3 = SE3_b2d @ SE3_s2b
    return SE3
