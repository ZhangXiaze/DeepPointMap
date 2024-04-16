import torch
import torch.nn as nn
from torch import Tensor as Tensor
from typing import Literal, Tuple, List
from network.encoder.utils import Querier, Sampler, index_points, coordinate_distance, build_mlp


class SetAbstraction(nn.Module):


    def __init__(self,
                 npoint: int,
                 radius: float,
                 nsample: int,
                 in_channel: int,
                 sample: dict,
                 norm: Literal['bn', 'ln', 'in'] = 'ln',
                 bias: bool = True):

        assert 'type' in sample.keys(), f'key \'type\' must be in sample dict'
        assert sample['type'] in ['fps', 'voxel', 'fps-t3d'], f'{sample} is not a supported sampling way, ' \
                                                              f'please use \'fps\' or \'voxel\''
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.in_channel = in_channel
        self.mlp = build_mlp(in_channel=in_channel + 3, channel_list=[in_channel * 2], dim=2, norm=norm, bias=bias)
        sample_type = sample['type']
        if sample_type == 'voxel':
            assert 'size' in sample.keys() and 'range' in sample.keys()
            self.sample_kwargs = {'voxel_size': sample['size'], 'sample_range': sample['range']}
        else:
            self.sample_kwargs = {}
        self.sample = Sampler(sample_type)
        self.query = Querier('hybrid-t3d')

    def forward(self, points_coor: Tensor, points_fea: Tensor, points_padding: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        points_coor, points_fea = points_coor.permute(0, 2, 1), points_fea.permute(0, 2, 1)  # (B, C, N) -> (B, N, C)
        bs, nbr_point_in, _ = points_coor.shape
        num_point_out = self.npoint

        '''S'''
        new_coor, new_mask = self.sample(points=points_coor, points_padding=points_padding, K=num_point_out,
                                         **self.sample_kwargs)

        '''G'''
        group_idx = self.query(radius=self.radius, K=self.nsample, points=points_coor, centers=new_coor,
                               points_padding=points_padding)

        grouped_points_coor = index_points(points_coor[..., :3], group_idx)  #  (B, S, K, 3)
        grouped_points_coor -= new_coor[..., :3].view(bs, num_point_out, 1, 3)  #  (B, S, K, 3)
        grouped_points_coor = grouped_points_coor / self.radius  # (B, S, K, 3) 
        grouped_points_fea = index_points(points_fea, group_idx)  #  (B, S, K, C)
        grouped_points_fea = torch.cat([grouped_points_fea, grouped_points_coor], dim=-1)  #  (B, S, K, C+3)
        '''P'''
        # (B, S, K, C+3) -> (B, C+3, K, S) -mlp-> (B, D, K, S) -pooling-> (B, D, S)
        grouped_points_fea = grouped_points_fea.permute(0, 3, 2, 1)  # B, C_in+3, K, S  # 
        grouped_points_fea = self.mlp(grouped_points_fea)  # B, C_out, K, S
        new_fea = torch.max(grouped_points_fea, dim=2)[0]  # B, C_out, S

        new_coor = new_coor.permute(0, 2, 1)  # (B, 3, S)
        return new_coor, new_fea, new_mask


class LocalAggregation(nn.Module):


    def __init__(self,
                 radius: float,
                 nsample: int,
                 in_channel: int,
                 norm: Literal['bn', 'ln', 'in'] = 'ln',
                 bias: bool = True):

        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.in_channel = in_channel
        self.mlp = build_mlp(in_channel=in_channel + 3, channel_list=[in_channel], dim=2, norm=norm, bias=bias)
        self.query = Querier('hybrid-t3d')

    def forward(self, points_coor: Tensor, points_fea: Tensor, points_padding: Tensor) -> Tensor:


        '''reshape'''
        # (B, C, N) -> (B, N, C)
        points_coor, points_fea = points_coor.permute(0, 2, 1), points_fea.permute(0, 2, 1)
        bs, npoint, _ = points_coor.shape

        '''G'''
        #  (B, N, K)
        group_idx = self.query(radius=self.radius, K=self.nsample, points=points_coor, centers=points_coor,
                               points_padding=points_padding)

        grouped_points_coor = index_points(points_coor[..., :3], group_idx)  #  (B, N, K, 3)
        grouped_points_coor = grouped_points_coor - points_coor[..., :3].view(bs, npoint, 1, 3) 
        grouped_points_coor = grouped_points_coor / self.radius  # 
        grouped_points_fea = index_points(points_fea, group_idx)  #  (B, N, K, C)
        grouped_points_fea = torch.cat([grouped_points_fea, grouped_points_coor], dim=-1)  #  (B, N, K, C+3)

        '''P'''
        # (B, N, K, C+3) -> (B, C+3, K, N) -mlp-> (B, D, K, N) -pooling-> (B, D, N)
        grouped_points_fea = grouped_points_fea.permute(0, 3, 2, 1)  
        grouped_points_fea = self.mlp(grouped_points_fea)
        new_fea = torch.max(grouped_points_fea, dim=2)[0]

        return new_fea


class InvResMLP(nn.Module):


    def __init__(self,
                 radius: float,
                 nsample: int,
                 in_channel: int,
                 expansion: int = 4,
                 norm: Literal['bn', 'ln', 'in'] = 'ln',
                 bias: bool = True):

        super().__init__()
        self.la = LocalAggregation(radius=radius, nsample=nsample, in_channel=in_channel, norm=norm, bias=bias)
        channel_list = [in_channel * expansion, in_channel]
        self.pw_conv = build_mlp(in_channel=in_channel, channel_list=channel_list, dim=1, drop_last_act=True,
                                 norm=norm, bias=bias)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points: List[Tensor]) -> List[Tensor]:

        points_coor, points_fea, points_padding = points
        identity = points_fea
        points_fea = self.la(points_coor, points_fea, points_padding)
        points_fea = self.pw_conv(points_fea)
        points_fea = points_fea + identity
        points_fea = self.act(points_fea)
        return [points_coor, points_fea, points_padding]


class Stage(nn.Module):


    def __init__(self,
                 npoint: int,
                 radius_list: List[float],
                 nsample_list: List[int],
                 in_channel: int,
                 sample: dict,
                 expansion: int = 4,
                 norm: Literal['bn', 'ln', 'in'] = 'ln',
                 bias: bool = True):

        assert len(radius_list) == len(nsample_list)
        super().__init__()
        self.sa = SetAbstraction(npoint=npoint, radius=radius_list[0], nsample=nsample_list[0],
                                 in_channel=in_channel, sample=sample, norm=norm, bias=bias)
        irm = []
        for i in range(1, len(radius_list)):
            irm.append(
                InvResMLP(radius=radius_list[i], nsample=nsample_list[i], in_channel=in_channel * 2,
                          expansion=expansion, norm=norm, bias=bias)
            )
        if len(irm) > 0:
            self.irm = nn.Sequential(*irm)
        else:
            self.irm = nn.Identity()

    def forward(self, points_coor: Tensor, points_fea: Tensor, points_padding: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        new_coor, new_fea, new_padding = self.sa(points_coor, points_fea, points_padding)
        new_coor, new_fea, new_padding = self.irm([new_coor, new_fea, new_padding])
        return new_coor, new_fea, new_padding


class FeaturePropagation(nn.Module):


    def __init__(self,
                 in_channel: List[int],
                 mlp: List[int],
                 norm: Literal['bn', 'ln', 'in'] = 'ln',
                 bias: bool = True):

        super(FeaturePropagation, self).__init__()
        self.mlp = build_mlp(in_channel=sum(in_channel), channel_list=mlp, dim=1, norm=norm, bias=bias)

    def forward(self, points_coor1: Tensor, points_coor2: Tensor, points_fea1: Tensor, points_fea2: Tensor,
                points_padding2: Tensor) -> Tensor:

        B, _, N = points_coor1.shape
        _, _, S = points_coor2.shape

        if S == 1:
            # (B, D2, 1) -> (B, D2, N)  
            new_fea = points_fea2.repeat(1, 1, N)
        else:
            # (B, C, N) -> (B, N, C)
            points_coor1, points_coor2, points_fea2 = \
                points_coor1.transpose(1, 2), points_coor2.transpose(1, 2), points_fea2.transpose(1, 2)

            points_coor2 = points_coor2.clone()
            points_coor2[points_padding2] = points_coor2.abs().max() * 3

            dists = coordinate_distance(points_coor1[..., :3], points_coor2[..., :3])
            dists, idx = torch.topk(dists, k=3, dim=-1, largest=False)  #  (B, N, 3)

            dist_recip = 1.0 / dists.clamp(min=1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points_fea2, idx) * weight.view(B, N, 3, 1), dim=2)
            # (B, N, D2) -> (B, D2, N)
            new_fea = interpolated_points.permute(0, 2, 1)

        #  (B, D2, N) -> (B, D1+D2, N) -> (B, D, N)
        new_fea = torch.cat((points_fea1, new_fea), dim=1)
        new_fea = self.mlp(new_fea)
        return new_fea
