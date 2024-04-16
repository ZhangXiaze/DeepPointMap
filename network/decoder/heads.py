import torch
import torch.nn as nn
from torch import Tensor as Tensor


def CoarsePairingHead(emb_dim: int):
    return nn.Sequential(
        nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=1)
    )


def SimilarityHead(emb_dim: int):
    return nn.Sequential(
        nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=1)
    )


class OffsetHead(nn.Module):
    def __init__(self, emb_dim: int, coor_dim: int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=emb_dim // 2, out_channels=emb_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=emb_dim // 4, out_channels=emb_dim // 8, kernel_size=1),
        )
        self.downsample = nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim // 8, kernel_size=1)
        self.head = nn.Conv1d(in_channels=emb_dim // 8, out_channels=coor_dim, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, pcd_fea: Tensor) -> Tensor:

        out = self.mlp(pcd_fea)
        identity = self.downsample(pcd_fea)
        out = self.act(out + identity)
        out = self.head(out)
        return out


class OverlapHead(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=1)
        )
        self.projection = nn.Sequential(
            nn.Linear(in_features=2 * emb_dim, out_features=2 * emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2 * emb_dim, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, src_fea, dst_fea) -> Tensor:
        src_fea = self.mlp(src_fea)
        dst_fea = self.mlp(dst_fea)

        # (B, C, N) -> (B, C)
        src_fea = torch.mean(src_fea, dim=-1)
        dst_fea = torch.mean(dst_fea, dim=-1)

        loop_pro = self.projection(torch.cat([src_fea, dst_fea], dim=-1)).flatten()  # (B,)
        return loop_pro
