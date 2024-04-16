import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as Tensor
from typing import Tuple


class DescriptorAttentionLayer(nn.Module):

    def __init__(self, emb_dim: int):

        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=8, batch_first=True, dropout=0)
        self.cross_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=8, batch_first=True, dropout=0)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=emb_dim, out_features=emb_dim))
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.norm3 = nn.LayerNorm(emb_dim)

    def forward(self, src_fea: Tensor, dst_fea: Tensor, src_pos_embedding: Tensor, dst_pos_embedding: Tensor,
                src_padding_mask=None, dst_padding_mask=None) -> Tuple[Tensor, Tensor]:
        # (B, C, N) -> (B, N, C)
        src_fea, dst_fea = src_fea.transpose(1, 2), dst_fea.transpose(1, 2)
        src_pos_embedding, dst_pos_embedding = src_pos_embedding.transpose(1, 2), dst_pos_embedding.transpose(1, 2)

        # self-attn
        src_fea = src_fea + src_pos_embedding
        dst_fea = dst_fea + dst_pos_embedding
        out_fea, _out_weight = self.self_attn(query=src_fea, key=src_fea, value=src_fea, key_padding_mask=src_padding_mask)
        src_fea = self.norm1(src_fea + out_fea)
        out_fea, _out_weight = self.self_attn(query=dst_fea, key=dst_fea, value=dst_fea, key_padding_mask=dst_padding_mask)
        dst_fea = self.norm1(dst_fea + out_fea)

        # xros-attn
        src_fea = src_fea + src_pos_embedding
        dst_fea = dst_fea + dst_pos_embedding
        src_out_fea, _src_out_weight = self.cross_attn(query=src_fea, key=dst_fea, value=dst_fea, key_padding_mask=dst_padding_mask)
        dst_out_fea, _dst_out_weight = self.cross_attn(query=dst_fea, key=src_fea, value=src_fea, key_padding_mask=src_padding_mask)
        src_fea = self.norm2(src_fea + src_out_fea)
        dst_fea = self.norm2(dst_fea + dst_out_fea)

        # mlp: add & norm
        src_fea = self.norm3(self.mlp(src_fea) + src_fea)
        dst_fea = self.norm3(self.mlp(dst_fea) + dst_fea)

        src_fea, dst_fea = src_fea.transpose(1, 2), dst_fea.transpose(1, 2)
        return src_fea, dst_fea


class PositionEmbeddingCoordsSine(nn.Module):

    def __init__(self, in_dim: int = 3, emb_dim: int = 256, temperature: int = 10000, scale: float = 1.0):

        super().__init__()

        self.in_channels = in_dim
        self.num_pos_feats = emb_dim // in_dim // 2 * 2
        self.temperature = temperature
        self.padding = emb_dim - self.num_pos_feats * self.in_channels
        self.scale = scale * math.pi

    def forward(self, coor: Tensor) -> Tensor:

        coor = coor.transpose(1, 2)
        assert coor.shape[-1] == self.in_channels

        dim_t = torch.arange(self.num_pos_feats, dtype=coor.dtype, device=coor.device)
        dim_t = self.temperature**(2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        coor = coor * self.scale
        pos_divided = coor.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*coor.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = F.pad(pos_emb, (0, self.padding))
        pos_emb = pos_emb.transpose(1, 2)
        return pos_emb
