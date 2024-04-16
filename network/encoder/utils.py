import torch
import torch.nn as nn
import numpy as np
from torch import Tensor as Tensor
from typing import Literal, Tuple, List
from random import randint
import colorlog as logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from pytorch3d.ops import ball_query, knn_points, knn_gather, sample_farthest_points
except:
    pass
print_t3d_warning = False


class Querier:

    def __init__(self, method: Literal['knn', 'hybrid', 'ball', 'knn-t3d', 'ball-t3d', 'hybrid-t3d']):
        method_dict = {
            'knn': self.knn_query,
            'ball': self.ball_query,
            'hybrid': self.hybrid_query,
            'knn-t3d': self.knn_query_t3d,
            'ball-t3d': self.ball_query_t3d,
            'hybrid-t3d': self.hybrid_query_t3d,
        }
        if method.endswith('t3d'):
            try:
                from pytorch3d.ops import ball_query, knn_points, knn_gather
            except:
                method = method[:-4]
                global print_t3d_warning
                if not print_t3d_warning:
                    logger.warning(f'Module pytorch3d not found. The implementations of python version will be used, '
                                   f'which may cause significant speed decrease.')
                    print_t3d_warning = True
        self.query_method = method_dict[method.lower()]

    def __call__(self, *args, **kwargs):
        grouped_idx = self.query_method(**kwargs)
        return grouped_idx

    @staticmethod
    def knn_query(K: int, points: Tensor, centers: Tensor, points_padding: Tensor) -> Tensor:

        points = points.clone()
        points[points_padding] = points.abs().max() * 3  

        dist = coordinate_distance(centers[..., :3], points[..., :3])
        group_idx = torch.topk(dist, k=K, dim=-1, largest=False)[1]  

        return group_idx

    @staticmethod
    def ball_query(radius: float, K: int, points: Tensor, centers: Tensor, points_padding: Tensor) -> Tensor:

        B, N, _ = points.shape
        _, S, _ = centers.shape
        device = points.device
        points = points.clone()
        points[points_padding] = points.abs().max() * 3  

        dist = coordinate_distance(centers[..., :3], points[..., :3])  
        group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
        group_idx[dist > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :K]
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, K])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]

        return group_idx

    @staticmethod
    def hybrid_query(radius: float, K: int, points: Tensor, centers: Tensor, points_padding: Tensor) -> Tensor:

        B, N, _ = points.shape
        _, S, _ = centers.shape
        points = points.clone()
        points[points_padding] = points.abs().max() * 3  

        dist = coordinate_distance(centers[..., :3], points[..., :3]) 
        dist, group_idx = torch.topk(dist, k=K, dim=-1, largest=False)
        mask = dist > (radius ** 2)  
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, K])
        group_idx[mask] = group_first[mask]

        return group_idx

    @staticmethod
    def knn_query_t3d(K: int, points: Tensor, centers: Tensor, points_padding: Tensor) -> Tensor:

        result = knn_points(p1=centers[..., :3], p2=points[..., :3], lengths2=(~points_padding).sum(1),
                            K=K, return_nn=False, return_sorted=False)
        idx = result.idx
        return idx

    @staticmethod
    def ball_query_t3d(radius: float, K: int, points: Tensor, centers: Tensor, points_padding: Tensor) -> Tensor:

        result = ball_query(p1=centers[..., :3], p2=points[..., :3], lengths2=(~points_padding).sum(1),
                            K=K, radius=radius, return_nn=False)

        idx = result.idx
        idx_mask = idx == -1
        if idx_mask.any():
            padding_idx = idx[:, :, :1].repeat(1, 1, K)
            idx[idx_mask] = padding_idx[idx_mask]
        return idx

    @staticmethod
    def hybrid_query_t3d(radius: float, K: int, points: Tensor, centers: Tensor, points_padding: Tensor) -> Tensor:

        result = knn_points(p1=centers[..., :3], p2=points[..., :3], lengths2=(~points_padding).sum(1),
                            K=K, return_nn=False, return_sorted=False)

        idx = result.idx
        dists = result.dists
        dists_mask = dists > (radius ** 2)
        padding_idx = idx[:, :, :1].repeat(1, 1, K)
        idx[dists_mask] = padding_idx[dists_mask]
        return idx


class Sampler:

    def __init__(self, method: Literal['fps', 'voxel', 'fps-t3d']):
        method_dict = {
            'fps': self.fps,
            'voxel': self.voxel,
            'fps-t3d': self.fps_t3d,
        }
        if method.endswith('t3d'):
            try:
                from pytorch3d.ops import sample_farthest_points
            except:
                method = method[:-4]
                global print_t3d_warning
                if not print_t3d_warning:
                    logger.warning(f'Module pytorch3d not found. The implementations of python version will be used, '
                                   f'which may cause significant speed decrease.')
                    print_t3d_warning = True
        self.sample_method = method_dict[method.lower()]

    def __call__(self, *args, **kwargs):
        return self.sample_method(**kwargs)

    @staticmethod
    def voxel(points: Tensor, points_padding: Tensor, K: int, voxel_size: float = 0.3, sample_range: float = 1.0) \
            -> Tuple[Tensor, Tensor]:

        B, device = points.shape[0], points.device
        pcd_xyz = points[:, :, :3].clone()

        pcd_xyz[points_padding] = 2 * sample_range

        xyz_min = torch.min(pcd_xyz, dim=1)[0]
        xyz_max = torch.max(pcd_xyz, dim=1)[0]
        X, Y, Z = torch.div(xyz_max[:, 0] - xyz_min[:, 0], voxel_size, rounding_mode='trunc') + 1, \
                  torch.div(xyz_max[:, 1] - xyz_min[:, 1], voxel_size, rounding_mode='trunc') + 1, \
                  torch.div(xyz_max[:, 2] - xyz_min[:, 2], voxel_size, rounding_mode='trunc') + 1

        dis_mask = torch.sum(pcd_xyz.pow(2), dim=-1) <= (sample_range * sample_range) 

        X, Y = X.unsqueeze(1), Y.unsqueeze(1)
        relative_xyz = pcd_xyz - xyz_min.unsqueeze(1)
        voxel_xyz = torch.div(relative_xyz, voxel_size, rounding_mode='trunc').int()
        voxel_id = (voxel_xyz[:, :, 0] + voxel_xyz[:, :, 1] * X + voxel_xyz[:, :, 2] * X * Y).int()

        sampled_points = []
        dis = torch.sum((relative_xyz - voxel_xyz * voxel_size - voxel_size / 2).pow(2), dim=-1)

        dis, sorted_id = torch.sort(dis, dim=-1)
        b_id = torch.arange(points.shape[0], device=device).unsqueeze(1)
        voxel_id = voxel_id[b_id, sorted_id]
        points = points[b_id, sorted_id]

        dis_mask = dis_mask[b_id, sorted_id]

        for b in range(points.shape[0]):

            b_voxel_id = voxel_id[b, dis_mask[b]]
            b_pcd = points[b, dis_mask[b]]

            _, unique_id, cnt = np.unique(b_voxel_id.detach().cpu(), return_index=True, return_counts=True)
            unique_id, cnt = torch.tensor(unique_id, device=device), torch.tensor(cnt, device=device)

            if K is not None and unique_id.shape[0] > K:
                _, cnt_topk_id = torch.topk(cnt, k=K)
                unique_id = unique_id[cnt_topk_id]
            sampled_points.append(b_pcd[unique_id])

        if K is not None:
            padding_mask = torch.zeros(size=(B, K), dtype=torch.bool, device=device)
            for i, b_pcd in enumerate(sampled_points):
                if b_pcd.shape[0] < K:
                    zero_padding = torch.zeros(size=(K - b_pcd.shape[0], b_pcd.shape[1]),
                                               device=b_pcd.device, dtype=b_pcd.dtype)
                    sampled_points[i] = torch.cat((b_pcd, zero_padding), dim=0)
                    padding_mask[i, b_pcd.shape[0]:] = True 
        else:
            assert len(sampled_points) == 1
            padding_mask = torch.zeros(size=(1, sampled_points[0].shape[0]), dtype=torch.bool, device=device)

        sampled_points = torch.stack(sampled_points, dim=0)
        return sampled_points, padding_mask

    @staticmethod
    def fps(points: Tensor, points_padding: Tensor, K: int, random_start_point: bool = False) -> Tuple[Tensor, Tensor]:

        lengths = (~points_padding).sum(1)
        points_xyz = points[..., :3].clone()
        N, P, D = points_xyz.shape
        device = points_xyz.device

        if lengths is None:
            lengths = torch.full((N,), P, dtype=torch.int64, device=device)
        else:
            if lengths.shape != (N,):
                raise ValueError("points and lengths must have same batch dimension.")
            if lengths.max() > P:
                raise ValueError("Invalid lengths.")
        K = torch.full((N,), K, dtype=torch.int64, device=device)

        # Find max value of K
        max_K = K.max()

        # List of selected indices from each batch element
        all_sampled_indices = []

        for n in range(N):
            # Initialize an array for the sampled indices, shape: (max_K,)
            sample_idx_batch = torch.full(
                (max_K,),
                fill_value=-1,
                dtype=torch.int64,
                device=device,
            )

            closest_dists = points_xyz.new_full(
                (lengths[n],),
                float("inf"),
                dtype=torch.float32,
            )

            # Select a random point index and save it as the starting point
            selected_idx = randint(0, lengths[n] - 1) if random_start_point else 0
            sample_idx_batch[0] = selected_idx

            k_n = min(lengths[n], K[n])

            # Iteratively select points for a maximum of k_n
            for i in range(1, k_n):
                dist = points_xyz[n, selected_idx, :] - points_xyz[n, : lengths[n], :]
                dist_to_last_selected = (dist ** 2).sum(-1)  # (P - i)
                closest_dists = torch.min(dist_to_last_selected, closest_dists)  # (P - i)
                selected_idx = torch.argmax(closest_dists)
                sample_idx_batch[i] = selected_idx

            # Add the list of points for this batch to the final list
            all_sampled_indices.append(sample_idx_batch)

        all_sampled_indices = torch.stack(all_sampled_indices, dim=0)

        # Gather the points
        sampled_points = masked_gather(points, all_sampled_indices)

        padding_mask = all_sampled_indices < 0
        return sampled_points, padding_mask

    @staticmethod
    def fps_t3d(points: Tensor, points_padding: Tensor, K: int, random_start_point: bool = False) \
            -> Tuple[Tensor, Tensor]:

        if points.shape[-1] > 3:
            points_xyz = points[..., :3]
            idx = sample_farthest_points(points=points_xyz, lengths=(~points_padding).sum(1), K=K,
                                         random_start_point=random_start_point)[1]
            sampled_points = masked_gather(points, idx)
        else:
            sampled_points, idx = sample_farthest_points(points=points, lengths=(~points_padding).sum(1), K=K,
                                                         random_start_point=random_start_point)
        padding_mask = idx < 0
        return sampled_points, padding_mask


def coordinate_distance(src: Tensor, dst: Tensor) -> Tensor:

    B, M, _ = src.shape
    _, N, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, -1).view(B, M, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, N)
    return dist


def masked_gather(points: Tensor, idx: Tensor) -> Tensor:
    """
    Helper function for torch.gather to collect the points at
    the given indices in idx where some of the indices might be -1 to
    indicate padding. These indices are first replaced with 0.
    Then the points are gathered after which the padded values
    are set to 0.0.

    Args:
        points: (N, P, D) float32 tensor of points
        idx: (N, K) or (N, P, K) long tensor of indices into points, where
            some indices are -1 to indicate padding

    Returns:
        selected_points: (N, K, D) float32 tensor of points
            at the given indices
    """

    if len(idx) != len(points):
        raise ValueError("points and idx must have the same batch dimension")

    N, P, D = points.shape

    if idx.ndim == 3:
        # Case: KNN, Ball Query where idx is of shape (N, P', K)
        # where P' is not necessarily the same as P as the
        # points may be gathered from a different pointcloud.
        K = idx.shape[2]
        # Match dimensions for points and indices
        idx_expanded = idx[..., None].expand(-1, -1, -1, D)
        points = points[:, :, None, :].expand(-1, -1, K, -1)
    elif idx.ndim == 2:
        # Farthest point sampling where idx is of shape (N, K)
        idx_expanded = idx[..., None].expand(-1, -1, D)
    else:
        raise ValueError("idx format is not supported %s" % repr(idx.shape))

    idx_expanded_mask = idx_expanded.eq(-1)
    idx_expanded = idx_expanded.clone()
    # Replace -1 values with 0 for gather
    idx_expanded[idx_expanded_mask] = 0
    # Gather points
    selected_points = points.gather(dim=1, index=idx_expanded)
    # Replace padded values
    selected_points[idx_expanded_mask] = 0.0
    return selected_points


def index_points(points: Tensor, idx: Tensor) -> Tensor:

    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def build_mlp(in_channel: int, channel_list: List[int], dim: int = 2, bias: bool = False, drop_last_act: bool = False,
              norm: Literal['bn', 'ln', 'in'] = 'bn', act: Literal['relu', 'elu'] = 'relu') -> nn.Sequential:

    norm_1d = {'bn': nn.BatchNorm1d,
               'in': nn.InstanceNorm1d,
               'ln': LayerNorm1d}
    norm_2d = {'bn': nn.BatchNorm2d,
               'in': nn.InstanceNorm2d,
               'ln': LayerNorm2d}
    acts = {'relu': nn.ReLU,
            'elu': nn.ELU}

    if dim == 1:
        Conv = nn.Conv1d
        NORM = norm_1d.get(norm.lower(), nn.BatchNorm1d)
    else:
        Conv = nn.Conv2d
        NORM = norm_2d.get(norm.lower(), nn.BatchNorm2d)
    ACT = acts.get(act.lower(), nn.ReLU)

    mlp = []
    for channel in channel_list:
        # conv-norm-act
        mlp.append(Conv(in_channels=in_channel, out_channels=channel, kernel_size=1, bias=bias))
        mlp.append(NORM(channel))
        mlp.append(ACT(inplace=True))
        in_channel = channel

    if drop_last_act:
        mlp = mlp[:-1]

    return nn.Sequential(*mlp)


class LayerNorm1d(nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.ln = nn.LayerNorm(channel)

    def forward(self, x):
        """(B, C, N)"""
        out = self.ln(x.transpose(1, 2)).transpose(1, 2)
        return out


class LayerNorm2d(nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.ln = nn.LayerNorm(channel)

    def forward(self, x):
        """(B, C, H, W)"""
        out = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out
