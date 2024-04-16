import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as Tensor
from typing import List, Union, Tuple
from network.decoder.descriptor_attention import DescriptorAttentionLayer, PositionEmbeddingCoordsSine
from network.decoder.heads import CoarsePairingHead, SimilarityHead, OffsetHead, OverlapHead


class Decoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.decoder_cfg = self.args.decoder

        self.in_channel = self.decoder_cfg.in_channel
        self.model_channel = self.decoder_cfg.model_channel
        self.attention_layers = self.decoder_cfg.attention_layers

        self.tau = self.args.loss.tau
        self.descriptor_pairing_method = self._descriptor_pairing
        # ======================== Build Network ========================
        self.projection = nn.Conv1d(in_channels=self.in_channel, out_channels=self.model_channel, kernel_size=1)
        self.pos_embedding_layer = PositionEmbeddingCoordsSine(in_dim=3, emb_dim=self.model_channel)
        self.descriptor_attention = nn.ModuleList()
        for _ in range(self.attention_layers):
            self.descriptor_attention.append(DescriptorAttentionLayer(emb_dim=self.model_channel))
        self.similarity_head = SimilarityHead(emb_dim=self.model_channel)
        self.offset_head = OffsetHead(emb_dim=self.model_channel * 2)
        self.loop_head = OverlapHead(emb_dim=self.model_channel)
        self.coarse_pairing_head = CoarsePairingHead(emb_dim=self.in_channel)

    def forward(self, src_descriptor: Tensor, dst_descriptor: Tensor,
                src_padding_mask: Tensor = None, dst_padding_mask: Tensor = None,
                gt_Rt: Tuple[Tensor, Tensor] = None) -> List[Tensor]:
        assert self.training, 'forward is not available during inference!'
        return self._train_forward(src_descriptor, dst_descriptor, src_padding_mask, dst_padding_mask, gt_Rt)

    def _train_forward(self, src_descriptor: Tensor, dst_descriptor: Tensor,
                       src_padding_mask: Tensor, dst_padding_mask: Tensor,
                       gt_Rt: Tuple[Tensor, Tensor]) -> List[Tensor]:
        assert gt_Rt is not None, 'gt_Rt must be provided during training'
        self.gt_R, self.gt_T = gt_Rt

        '''unified descriptor -> coarse pairing feature'''
        src_fea, dst_fea = src_descriptor[:, :-3, :], dst_descriptor[:, :-3, :]
        src_coarse_pairing_fea = self.coarse_pairing_head(src_fea)
        dst_coarse_pairing_fea = self.coarse_pairing_head(dst_fea)

        '''unified descriptor -> correlated descriptor'''
        src_corr_descriptor, dst_corr_descriptor = \
            self._descriptor_attention_forward(src_descriptor, dst_descriptor, src_padding_mask, dst_padding_mask)
        src_fea, src_xyz = src_corr_descriptor[:, :-3, :], src_corr_descriptor[:, -3:, :]
        dst_fea, dst_xyz = dst_corr_descriptor[:, :-3, :], dst_corr_descriptor[:, -3:, :]

        '''correlated descriptors -> similarity feature'''
        src_pairing_fea = self.similarity_head(src_fea)
        dst_pairing_fea = self.similarity_head(dst_fea)

        '''correlated descriptors -> offset'''
        src_gt_xyz = self.gt_R @ src_xyz + self.gt_T
        src_gt_xyz, dst_gt_xyz = src_gt_xyz.transpose(1, 2), dst_xyz.transpose(1, 2)  # (B, M/N, 3)
        dist = torch.sum(torch.square(src_gt_xyz.unsqueeze(2) - dst_gt_xyz.unsqueeze(1)), dim=-1)  # (B, M, N)
        dist_mask = dist <= (self.args.loss.eps_offset ** 2)
        dist_mask &= ~src_padding_mask.unsqueeze(2)
        dist_mask &= ~dst_padding_mask.unsqueeze(1)

        pairs_index = torch.nonzero(dist_mask)
        batch_index = pairs_index[:, 0]
        src_index = pairs_index[:, 1]
        dst_index = pairs_index[:, 2]
        src_pair_gt_xyz = src_gt_xyz[batch_index, src_index, :]
        dst_pair_gt_xyz = dst_gt_xyz[batch_index, dst_index, :]
        src_pair_fea = src_fea.transpose(1, 2)[batch_index, src_index, :]
        dst_pair_fea = dst_fea.transpose(1, 2)[batch_index, dst_index, :]

        src2dst_R = torch.cat([self.gt_R[i].unsqueeze(0).repeat(num, 1, 1)
                               for i, num in enumerate(torch.sum(dist_mask, dim=[1, 2]))], dim=0)
        src_offset_gt = src2dst_R.transpose(1, 2) @ (dst_pair_gt_xyz - src_pair_gt_xyz).unsqueeze(2)
        dst_offset_gt = (src_pair_gt_xyz - dst_pair_gt_xyz).unsqueeze(2)
        src_offset_fea = torch.cat([src_pair_fea, dst_pair_fea], dim=1).unsqueeze(2)  # (K, C, 1)
        dst_offset_fea = torch.cat([dst_pair_fea, src_pair_fea], dim=1).unsqueeze(2)
        src_offset = self.offset_head(src_offset_fea)  # (K, 3, 1)
        dst_offset = self.offset_head(dst_offset_fea)
        src_offset_res, dst_offset_res = src_offset - src_offset_gt, dst_offset - dst_offset_gt
        return [src_pairing_fea, dst_pairing_fea,
                src_coarse_pairing_fea, dst_coarse_pairing_fea,
                src_offset_res, dst_offset_res]

    def registration_forward(self, src_descriptor: Tensor, dst_descriptor: Tensor,
                             src_padding_mask: Tensor = None, dst_padding_mask: Tensor = None,
                             num_sample: Union[int, float] = 0.5)\
            -> Tuple[Tensor, Tensor, Tensor, Union[List[float], float]]:

        if src_descriptor.ndim == 2 and dst_descriptor.ndim == 2:
            batch = False
            src_descriptor = src_descriptor.unsqueeze(0)
            dst_descriptor = dst_descriptor.unsqueeze(0)
        else:
            batch = True

        '''unified descriptor -> correlated descriptor'''
        src_corr_descriptor, dst_corr_descriptor = \
            self._descriptor_attention_forward(src_descriptor, dst_descriptor, src_padding_mask, dst_padding_mask)

        '''pairing correlated descriptor -> merge offset -> pairing coordinate'''
        src_pairing_descriptor, dst_pairing_descriptor, pairing_conf = \
            self.descriptor_pairing_method(src_corr_descriptor, dst_corr_descriptor, num_sample)
        src_pairing_coor, dst_pairing_coor, pairing_conf = \
            self._get_corres_sets(src_pairing_descriptor, dst_pairing_descriptor, pairing_conf)

        '''solve transformation with SVD'''
        Rs, Ts, inlier_mask_list, inlier_rmse_list = \
            self._solve_transformation_SVD(pairing_conf, src_pairing_coor, dst_pairing_coor)

        if not batch:
            Rs = Rs[0]  # (3, 3)
            Ts = Ts[0]  # (3, 1)
            pairing_conf = pairing_conf[0, inlier_mask_list[0]]  # (K,)
            inlier_rmse_list = inlier_rmse_list[0]  # float
        else:
            Rs = Rs  # (1, 3, 3)
            Ts = Ts  # (1, 3, 1)
            pairing_conf = pairing_conf[0, inlier_mask_list[0]].unsqueeze(0)  # (1, K)
            inlier_rmse_list = inlier_rmse_list  # [float]
        return Rs, Ts, pairing_conf, inlier_rmse_list

    def loop_detection_forward(self, src_descriptor: Tensor, dst_descriptor: Tensor,
                               src_padding_mask: Tensor = None, dst_padding_mask: Tensor = None):
        if src_descriptor.ndim == 2 and dst_descriptor.ndim == 2:
            src_descriptor = src_descriptor.unsqueeze(0)
            dst_descriptor = dst_descriptor.unsqueeze(0)

        '''unified descriptor -> correlated descriptor'''
        src_corr_descriptor, dst_corr_descriptor = \
            self._descriptor_attention_forward(src_descriptor, dst_descriptor, src_padding_mask, dst_padding_mask)

        '''correlated descriptor -> loop detection'''
        src_fea, dst_fea = src_corr_descriptor[:, :-3, :], dst_corr_descriptor[:, :-3, :]
        loop_pro = self.loop_head(src_fea, dst_fea)

        return loop_pro

    def _descriptor_attention_forward(self, src_descriptor: Tensor, dst_descriptor: Tensor,
                                      src_padding_mask: Tensor = None, dst_padding_mask: Tensor = None)\
            -> Tuple[Tensor, Tensor]:
        """extract correlated descriptors"""
        src_fea, src_xyz = src_descriptor[:, :-3, :], src_descriptor[:, -3:, :]
        dst_fea, dst_xyz = dst_descriptor[:, :-3, :], dst_descriptor[:, -3:, :]
        src_pos_embedding = self.pos_embedding_layer(src_xyz)
        dst_pos_embedding = self.pos_embedding_layer(dst_xyz)
        src_fea = self.projection(src_fea)
        dst_fea = self.projection(dst_fea)

        for layer in self.descriptor_attention:
            src_fea, dst_fea = layer(src_fea=src_fea, dst_fea=dst_fea,
                                     src_pos_embedding=src_pos_embedding, dst_pos_embedding=dst_pos_embedding,
                                     src_padding_mask=src_padding_mask, dst_padding_mask=dst_padding_mask)
        src_corr_descriptor = torch.cat([src_fea, src_xyz], dim=1)
        dst_corr_descriptor = torch.cat([dst_fea, dst_xyz], dim=1)
        return src_corr_descriptor, dst_corr_descriptor

    def _descriptor_pairing(self, src_corr_descriptor: Tensor, dst_corr_descriptor: Tensor,
                            num_sample: Union[int, float] = 0.5) -> Tuple[Tensor, Tensor, Tensor]:
        """descriptor match"""
        B, _, M = src_corr_descriptor.shape
        N, device = dst_corr_descriptor.shape[-1], src_corr_descriptor.device
        assert B == 1, 'batch size in inference must be 1'
        if isinstance(num_sample, int):
            num_sample = num_sample
        elif isinstance(num_sample, float) and num_sample > 1:
            num_sample = int(num_sample)
        elif isinstance(num_sample, float) and 0 < num_sample <= 1:
            num_sample = int(num_sample * (M + N))
        else:
            raise ValueError(f'Argument `num_sample` with value {num_sample} is not supported')
        num_sample = num_sample // 2 

        '''correlated descriptors -> similarity feature'''
        src_pairing_fea = self.similarity_head(src_corr_descriptor[:, :-3, :])
        dst_pairing_fea = self.similarity_head(dst_corr_descriptor[:, :-3, :])

        '''Sim Mat (B, M, N) -> top-k'''
        similarity_matrix = F.normalize(src_pairing_fea.transpose(1, 2), p=2, dim=2) @ F.normalize(dst_pairing_fea, p=2, dim=1)
        row_softmax = F.softmax(similarity_matrix / self.tau, dim=2)
        col_softmax = F.softmax(similarity_matrix / self.tau, dim=1)
        similarity_matrix = row_softmax * col_softmax
        similarity_matrix = similarity_matrix.reshape(B, M * N)
        k_pair_value, k_self_index = torch.topk(similarity_matrix, k=num_sample, dim=1)
        src_pairing_index = k_self_index // N 
        dst_pairing_index = k_self_index % N  

        '''top-k -> Sort'''
        src_pairing_index, dst_pairing_index = src_pairing_index.squeeze(0), dst_pairing_index.squeeze(0)
        src_pairing_descriptor = src_corr_descriptor[:, :, src_pairing_index]  # (1, C, K)
        dst_pairing_descriptor = dst_corr_descriptor[:, :, dst_pairing_index]  # (1, C, K)
        pairing_conf = k_pair_value  # (1, K)

        return src_pairing_descriptor, dst_pairing_descriptor, pairing_conf

    def _get_corres_sets(self, src_pairing_descriptor: Tensor, dst_pairing_descriptor: Tensor, pairing_conf: Tensor)\
            -> Tuple[Tensor, Tensor, Tensor]:
        pairing_fea_s2d = torch.cat([src_pairing_descriptor[:, :-3, :], dst_pairing_descriptor[:, :-3, :]], dim=1)
        pairing_fea_d2s = torch.cat([dst_pairing_descriptor[:, :-3, :], src_pairing_descriptor[:, :-3, :]], dim=1)
        pairing_offset_s2d = self.offset_head(pairing_fea_s2d)
        pairing_offset_d2s = self.offset_head(pairing_fea_d2s)
        # pairing_offset_s2d[:, :, :] = 0  # if hide offset prediction
        # pairing_offset_d2s[:, :, :] = 0

        src_align_coor = src_pairing_descriptor[:, -3:, :] + pairing_offset_s2d
        dst_align_coor = dst_pairing_descriptor[:, -3:, :] + pairing_offset_d2s
        src_pairing_coor = torch.cat([src_align_coor, src_pairing_descriptor[:, -3:, :]], dim=-1)
        dst_pairing_coor = torch.cat([dst_pairing_descriptor[:, -3:, :], dst_align_coor], dim=-1)
        pairing_conf = pairing_conf.repeat(1, 2)  # (1, 2K)

        outlier_max = self.args.loss.eps_offset ** 2
        offset_outlier_mask_s2d = torch.sum(torch.square(pairing_offset_s2d), dim=1) <= outlier_max
        offset_outlier_mask_d2s = torch.sum(torch.square(pairing_offset_d2s), dim=1) <= outlier_max
        offset_outlier_mask = torch.cat([offset_outlier_mask_s2d.squeeze(0), offset_outlier_mask_d2s.squeeze(0)], dim=0)
        src_pairing_coor = src_pairing_coor[..., offset_outlier_mask]
        dst_pairing_coor = dst_pairing_coor[..., offset_outlier_mask]
        pairing_conf = pairing_conf[..., offset_outlier_mask]

        return src_pairing_coor, dst_pairing_coor, pairing_conf

    @staticmethod
    def _solve_transformation_SVD(pairing_conf: Tensor, src_pairing_coor: Tensor, dst_pairing_coor: Tensor,
                                  num_iter: int = 3, std_ratio: float = 3.0) -> Tuple[Tensor, Tensor, List, List]:
        R_list, T_list, inlier_mask_list, inlier_rmse_list = [], [], [], []
        for weight, src, dst in zip(pairing_conf, src_pairing_coor, dst_pairing_coor):
            iter_cnt = 0
            inlier_mask = weight > 0.5 
            _, ids = torch.topk(weight, k=min(64, len(weight)), dim=0)
            inlier_mask[ids] = True
            while True:
                src_inner, dst_inner, weight_inner = src[:, inlier_mask], dst[:, inlier_mask], weight[inlier_mask]  # (3/1, 2K)
                src_xyz_inner_centroid = (src_inner * weight_inner).sum(dim=1, keepdim=True) / weight_inner.sum()  # (3, 1)
                dst_xyz_inner_centroid = (dst_inner * weight_inner).sum(dim=1, keepdim=True) / weight_inner.sum()  # (3, 1)
                S = (src_inner - src_xyz_inner_centroid) @ torch.diag(weight_inner) @ (dst_inner - dst_xyz_inner_centroid).T  # (3, 3)

                u, s, v = torch.svd(S.double())  # S should be FP64
                R = v @ u.T
                T = dst_xyz_inner_centroid.double() - R @ src_xyz_inner_centroid.double()
                R, T = R.float(), T.float()

                err = torch.norm(R @ src + T - dst, p=2, dim=0)
                inlier_mean, inlier_std = err[inlier_mask].mean(), err[inlier_mask].std()
                new_inlier = (err <= (inlier_mean + std_ratio * inlier_std))

                iter_cnt += 1
                if iter_cnt >= num_iter or (inlier_mask == new_inlier).all() or new_inlier.sum() < 30:
                    inlier_mask = new_inlier
                    break
                else:
                    inlier_mask = new_inlier
            R_list.append(R)
            T_list.append(T)
            inlier_mask_list.append(inlier_mask)
            inlier_rmse = (R @ src[:, inlier_mask] + T - dst[:, inlier_mask]).pow(2).sum(0).mean().sqrt().item()
            inlier_rmse_list.append(inlier_rmse)

        Rs = torch.stack(R_list, dim=0)  # [B, 3, 3]
        Ts = torch.stack(T_list, dim=0)  # [B, 3, 1]
        return Rs, Ts, inlier_mask_list, inlier_rmse_list
