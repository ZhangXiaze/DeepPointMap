import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor as Tensor
from typing import Tuple
import colorlog as logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RegistrationLoss(nn.Module):
    """
    L = lambda1 * L_p + lambda2 * L_c + lambda3 * L_o
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.loss_cfg = self.args.loss

        self.tau = self.loss_cfg.tau
        self.offset_value = self.loss_cfg.offset_value
        self.eps_positive = self.loss_cfg.eps_positive
        self.eps_offset = self.loss_cfg.eps_offset
        self.lambda_p = self.loss_cfg.lambda_p
        self.lambda_c = self.loss_cfg.lambda_c
        self.lambda_o = self.loss_cfg.lambda_o

    def forward(self, src_global_coor: Tensor, dst_global_coor: Tensor,
                src_padding_mask: Tensor, dst_padding_mask: Tensor,
                src_pairing_fea: Tensor, dst_pairing_fea: Tensor,
                src_coarse_pairing_fea: Tensor, dst_coarse_pairing_fea: Tensor,
                src_offset_res: Tensor, dst_offset_res: Tensor):
        """
        :param src_global_coor: (B, 3, S) 
        :param dst_global_coor: (B, 3, D) 
        :param src_padding_mask: (B, S) 
        :param dst_padding_mask: (B, D) 
        :param src_pairing_fea: (B, C, S) 
        :param dst_pairing_fea: (B, C, D) 
        :param src_coarse_pairing_fea: (B, C', S) 
        :param dst_coarse_pairing_fea: (B, C', D) 
        :param src_offset_res: (K, 3, 1)
        :param dst_offset_res: (K, 3, 1)
        :return: loss, metrics
        """
        src_padding_mask, dst_padding_mask = ~src_padding_mask, ~dst_padding_mask
        src_global_coor, dst_global_coor = src_global_coor.transpose(1, 2), dst_global_coor.transpose(1, 2)
        src_pairing_fea, dst_pairing_fea = src_pairing_fea.transpose(1, 2), dst_pairing_fea.transpose(1, 2)
        src_coarse_pairing_fea, dst_coarse_pairing_fea = \
            src_coarse_pairing_fea.transpose(1, 2), dst_coarse_pairing_fea.transpose(1, 2)
        src_offset_res, dst_offset_res = src_offset_res.transpose(1, 2), dst_offset_res.transpose(1, 2)

        '''src ==> dst'''
        corr_ids_src, corr_mask_src, neutral_mask_src = \
            self.make_pairs(src_global_coor, dst_global_coor, self.eps_positive)
        loss_pairing_src = self.pairing_loss(
            src_pairing_fea, dst_pairing_fea, src_padding_mask, corr_ids_src, corr_mask_src,
            neutral_mask=torch.zeros_like(neutral_mask_src, dtype=torch.bool, device=neutral_mask_src.device)
        )
        loss_coarse_pairing_src = self.pairing_loss(
            src_coarse_pairing_fea, dst_coarse_pairing_fea, src_padding_mask, corr_ids_src, corr_mask_src,
            neutral_mask=neutral_mask_src
        )
        loss_offset_src = self.offset_loss(src_offset_res)
        top1_pairing_acc_src = self.eval_pairing_acc(
            src_pairing_fea, dst_pairing_fea, src_padding_mask, corr_ids_src, corr_mask_src)

        '''dst ==> src'''
        corr_ids_dst, corr_mask_dst, neutral_mask_dst = \
            self.make_pairs(dst_global_coor, src_global_coor, self.eps_positive)
        loss_pairing_dst = self.pairing_loss(
            dst_pairing_fea, src_pairing_fea, dst_padding_mask, corr_ids_dst, corr_mask_dst,
            neutral_mask=torch.zeros_like(neutral_mask_dst, dtype=torch.bool, device=neutral_mask_dst.device)
        )
        loss_coarse_pairing_dst = self.pairing_loss(
            dst_coarse_pairing_fea, src_coarse_pairing_fea, dst_padding_mask, corr_ids_dst, corr_mask_dst,
            neutral_mask=neutral_mask_dst
        )
        loss_offset_dst = self.offset_loss(dst_offset_res)
        top1_pairing_acc_dst = self.eval_pairing_acc(
            dst_pairing_fea, src_pairing_fea, dst_padding_mask, corr_ids_dst, corr_mask_dst)

        '''loss = forward + backward'''
        loss_pairing = (loss_pairing_src + loss_pairing_dst) / 2
        loss_coarse_pairing = (loss_coarse_pairing_src + loss_coarse_pairing_dst) / 2
        loss_offset = (loss_offset_src + loss_offset_dst) / 2
        top1_pairing_acc = (top1_pairing_acc_src + top1_pairing_acc_dst) / 2
        loss = self.lambda_p * loss_pairing + self.lambda_c * loss_coarse_pairing + self.lambda_o * loss_offset

        return loss, top1_pairing_acc, loss_pairing, loss_coarse_pairing, loss_offset

    @staticmethod
    def make_pairs(src_global_coor: Tensor, dst_global_coor: Tensor, dis_threshold: float)\
            -> Tuple[Tensor, Tensor, Tensor]:

        B, S, device = src_global_coor.shape[0], src_global_coor.shape[1], src_global_coor.device

        dis = torch.sum(torch.square(src_global_coor.unsqueeze(2) - dst_global_coor.unsqueeze(1)), dim=-1)

        min_dis, min_dis_corr_ids = torch.min(dis, dim=-1)  # (B, S)

        batch_idx = torch.arange(0, B, device=device).unsqueeze(1).repeat(1, S)
        row_idx = torch.arange(0, S, device=device).unsqueeze(0).repeat(B, 1)
        neutral_mask = dis <= (dis_threshold * dis_threshold)  # (B, S, D)
        neutral_mask[batch_idx, row_idx, min_dis_corr_ids] = False

        corr_mask = min_dis <= (dis_threshold * dis_threshold)  # (B, S)
        corr_ids = min_dis_corr_ids
        corr_ids[~corr_mask] = -1

        return corr_ids, corr_mask, neutral_mask

    def pairing_loss(self, src_pairing_fea: Tensor, dst_pairing_fea: Tensor, src_padding_mask: Tensor,
                     corr_ids: Tensor, corr_mask: Tensor, neutral_mask: Tensor) -> Tensor:
        B, S, D = src_pairing_fea.shape[0], src_pairing_fea.shape[1], dst_pairing_fea.shape[1]

        samples_src = F.normalize(src_pairing_fea, p=2, dim=-1)
        samples_dst = F.normalize(dst_pairing_fea, p=2, dim=-1)

        logits = samples_src @ samples_dst.transpose(1, 2)

        src_padding_mask_1d = src_padding_mask.reshape(-1)
        logits = logits.reshape(B * S, D)[src_padding_mask_1d]
        labels = corr_ids.reshape(-1)[src_padding_mask_1d]
        neutral_mask = neutral_mask.reshape(B * S, D)[src_padding_mask_1d]
        corr_mask = corr_mask.reshape(-1)[src_padding_mask_1d]

        logits[neutral_mask] = -1e8
        logits_pos = logits[corr_mask]
        labels = labels[corr_mask]

        # InfoNCE
        if logits_pos.shape[0] > 0:
            logits_pos = logits_pos / self.tau
            logprobs_pos = F.log_softmax(logits_pos, dim=-1) 
            loss_pos = -logprobs_pos.gather(dim=-1, index=labels.unsqueeze(1)).squeeze(1) 
            loss_pos = loss_pos.mean()
        else:
            loss_pos = 0

        loss_pairing = loss_pos
        return loss_pairing

    def offset_loss(self, src_offset_res: Tensor) -> Tensor:
        src_offset_res = src_offset_res.squeeze(1)  # (K, 1, 3) -> (K, 3)
        if self.offset_value == 'manhattan':
            offset_err = torch.sum(torch.abs(src_offset_res), dim=-1) 
        elif self.offset_value == 'euclidean':
            offset_err = torch.norm(src_offset_res, p=2, dim=-1)
        elif self.offset_value == 'mahalanobis':
            try:
                cov_inv = torch.linalg.inv(torch.cov(src_offset_res.detach().T))
            except:
                cov_inv = torch.eye(3, device=src_offset_res.device, dtype=src_offset_res.dtype)
            # magic einsum, see `https://www.delftstack.com/howto/python/python-mahalanobis-distance/`
            offset_err = torch.sqrt(torch.einsum('nj,jk,nk->n', src_offset_res, cov_inv, src_offset_res))
        else:
            raise ValueError

        loss_offset = torch.sum(offset_err, dim=-1) / max(offset_err.shape[0], 1.0)
        return loss_offset

    @staticmethod
    def eval_pairing_acc(src_pairing_fea: Tensor, dst_pairing_fea: Tensor,
                         src_padding_mask: Tensor, corr_ids_src: Tensor, corr_mask_src: Tensor) -> float:
        samples_src = F.normalize(src_pairing_fea, p=2, dim=-1)
        samples_dst = F.normalize(dst_pairing_fea, p=2, dim=-1)
        similarity_matrix = samples_src @ samples_dst.transpose(1, 2)  # (B, S, D)
        _, src_corr_ids_pred = torch.max(similarity_matrix, dim=2)  # (B, S)

        src_padding_mask = src_padding_mask.reshape(-1)
        src_corr_ids_pred = src_corr_ids_pred.reshape(-1)[src_padding_mask]
        src_corr_ids_gt = corr_ids_src.reshape(-1)[src_padding_mask]
        corr_mask_src = corr_mask_src.reshape(-1)[src_padding_mask]

        correspondence_matrix = (src_corr_ids_pred == src_corr_ids_gt)[corr_mask_src]

        top1_acc = torch.sum(correspondence_matrix) / max(correspondence_matrix.shape[0], 1.0)
        return top1_acc.item()
