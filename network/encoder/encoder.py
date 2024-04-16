import torch.nn as nn
from torch import Tensor as Tensor
from typing import List
from network.encoder.pointnext import Stage, FeaturePropagation


class Encoder(nn.Module):
    """
    Backbone based on PointNeXt with FPN (for feature extraction)
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_cfg = self.args.encoder

        self.in_channel = self.encoder_cfg.in_channel
        self.out_channel = self.encoder_cfg.out_channel
        self.downsample_layers = len(self.encoder_cfg.npoint)  # 4
        self.upsample_layers = self.encoder_cfg.upsample_layers  # 2
        width = self.encoder_cfg.width
        norm = self.encoder_cfg.get('norm', 'LN').lower()
        bias = self.encoder_cfg.get('bias', True)

        # ======================== Build Network ========================
        self.point_mlp0 = nn.Conv1d(in_channels=self.in_channel, out_channels=width, kernel_size=1)
        self.downsampler = nn.ModuleList()
        self.upsampler = nn.ModuleList()
        for i in range(self.downsample_layers):  # [2048, 512, 128, 32]
            self.downsampler.append(
                Stage(npoint=self.encoder_cfg.npoint[i],
                      radius_list=self.encoder_cfg.radius_list[i],
                      nsample_list=self.encoder_cfg.nsample_list[i],
                      in_channel=width,
                      sample=self.encoder_cfg.sample[i],
                      expansion=self.encoder_cfg['expansion'],
                      norm=norm,
                      bias=bias))
            width *= 2

        upsampler_in = width
        for i in range(self.upsample_layers):
            upsampler_out = max(self.out_channel, width // 2)
            self.upsampler.append(
                FeaturePropagation(in_channel=[upsampler_in, width // 2],
                                   mlp=[upsampler_out, upsampler_out],
                                   norm=norm,
                                   bias=bias))
            width = width // 2
            upsampler_in = upsampler_out

    def forward(self, points: Tensor, points_padding: Tensor) -> List[Tensor]:
        l0_coor, l0_fea, l0_padding = points[:, :3, :].clone(), points[:, :self.in_channel, :].clone(), points_padding
        l0_fea = self.point_mlp0(l0_fea)  # (B, width, N)
        recorder = [[l0_coor, l0_fea, l0_padding]]

        # downsample
        for layer in self.downsampler:
            # (B, width, N0) -> (B, width * 2, N1) -> (B, width * 4, N2) ->...
            new_coor, new_fea, new_padding = layer(*recorder[-1])
            recorder.append([new_coor, new_fea, new_padding])

        # upsample
        for i, layer in enumerate(self.upsampler):
            points_coor1, points_fea1, points_padding1 = recorder[self.downsample_layers - i - 1]  # same layer
            points_coor2, points_fea2, points_padding2 = recorder[-1]  # deeper layer
            new_points_fea1 = layer(points_coor1, points_coor2, points_fea1, points_fea2, points_padding2)
            recorder.append([points_coor1.clone(), new_points_fea1, points_padding1.clone()])

        return recorder[-1]  # [coor, fea, padding]
