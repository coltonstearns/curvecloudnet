import torch
from torch_geometric.nn import fps, radius
from src.models.modules.point_conv import PointNetConv2
from torch_scatter import scatter_max, scatter_mean
from src.models.utils.point_ops import fps_pytorch3d, knn_ball_group_pytorch3d, knn_interpolate_pytorch3d, radius_1d_group_subset, knn_interpolate_1D_pytorch3d
from src.models.modules.fps_ops import CurveFPS, VoxelFPS

VISUALIZE_SA_GROUPINGS = False


class SAModuleSlow(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv2(nn, add_self_loops=False)

    def forward(self, x, pos, batch, point2curveidx=None, **kwargs):
        idx = fps(pos, batch, ratio=self.ratio).sort()[0]
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=128)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)

        # update forward state
        pos, batch = pos[idx], batch[idx]
        if point2curveidx is not None:
            point2curveidx = point2curveidx[idx]

        return x, pos, batch, point2curveidx


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn, k, curve_fps_arclen=None, voxel_size=None, downsample_type='random', attend_nn=None, aggr_type='max', normalize_radius=False, use_fast_knn=True, **kwargs):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.knn = k
        self.downsample_type = downsample_type
        self.use_fast_knn = use_fast_knn
        assert self.downsample_type in ['curve-fps', 'random', 'fps', 'voxel']
        self.curve_fps_arclen = curve_fps_arclen
        self.voxel_size = voxel_size
        self.normalize_radius = r if normalize_radius else None
        self.conv = PointNetConv2(nn, add_self_loops=False, aggr_type=aggr_type, attend_nn=attend_nn, normalize_radius=self.normalize_radius)

    def forward(self, x, pos, batch, point2curveidx=None, **kwargs):
        # downsample points
        if self.downsample_type == 'random':  # note: fix this to sample per-batch random
            num_idxs = int(pos.size(0) * self.ratio)
            idx = torch.randperm(pos.size(0))[:num_idxs]
            idx = torch.sort(idx)[0]
        elif self.downsample_type == 'curve-fps':
            fps_module = CurveFPS(self.curve_fps_arclen)
            idx = fps_module(pos.clone(), batch.clone(), point2curveidx.clone())
        elif self.downsample_type == 'voxel':
            fps_module = VoxelFPS(self.voxel_size)
            idx = fps_module(pos.clone(), batch.clone())
        else:
            idx = fps_pytorch3d(pos, batch, self.ratio)

        # Run ball grouping
        if self.use_fast_knn:
            row, col = knn_ball_group_pytorch3d(pos[idx].clone(), pos, batch[idx].clone(), batch, operation="knn", knn=self.knn, radius=self.r)
        else:
            row, col = knn_ball_group_pytorch3d(pos[idx].clone(), pos, batch[idx].clone(), batch, operation="ball-group", knn=self.knn, radius=self.r)
        edge_index = torch.stack([col, row], dim=0)  # row is queries, col is point idxs
        x_dst = None if x is None else x[idx]

        # Perform grouping convolution
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)

        # update forward state
        pos, batch = pos[idx], batch[idx]
        if point2curveidx is not None:
            point2curveidx = point2curveidx[idx]

        return x, pos, batch, point2curveidx


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn, **kwargs):
        super().__init__()
        self.nn = nn
        if "pooling" in kwargs:
            self.pooling = kwargs['pooling']
        else:
            self.pooling = "max"

    def forward(self, x, pos, batch, point2curveidx=None, **kwargs):
        # run PointNet with global max pool
        x = self.nn(torch.cat([x, pos], dim=1))
        size = int(batch.max().item() + 1)
        if self.pooling == "max":
            x, keypoint_idxs = scatter_max(x, batch, dim=0, out=None, dim_size=size)
        elif self.pooling == "mean":
            print("Mean Pooling! This is slower!")
            _, keypoint_idxs = scatter_max(x, batch, dim=0, out=None, dim_size=size)
            x = scatter_mean(x, batch, dim=0, out=None, dim_size=size)
        else:
            raise NotImplementedError("Pooling strategy %s not implemented!" % self.pooling)

        # compute "query idxs", ie the beginning of each batch
        assert torch.all((batch[1:] - batch[:-1]) >= 0)
        ptr = torch.where((batch[1:] - batch[:-1]) > 0)[0] + 1
        query_idxs = torch.cat([torch.zeros(1).to(keypoint_idxs), ptr])

        # get other state-tracking info
        pos = pos[query_idxs]
        batch = batch[query_idxs]

        # refine point2curveidx
        if point2curveidx is not None:
            point2curveidx = point2curveidx[query_idxs]

        return x, pos, batch, point2curveidx


class FPModule(torch.nn.Module):
    def __init__(self, k, nn, with_xyz=False):
        super().__init__()
        self.k = k
        self.nn = nn
        self.with_xyz = with_xyz

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip, point2curveidx=None,  point2curveidx_skip=None, **kwargs):
        # perform KNN interpolation
        x = knn_interpolate_pytorch3d(x, pos, pos_skip, batch, batch_skip, k=self.k)
        assert not torch.any(torch.isnan(x))

        # update interpolated feature
        if x_skip is not None and not self.with_xyz:
            x = torch.cat([x, x_skip.clone()], dim=1)
        elif x_skip is not None and self.with_xyz:
            x = torch.cat([x, x_skip.clone(), pos_skip.clone()[:, :3]], dim=1)
        elif x_skip is None and self.with_xyz:  # implies we are back to first layer
            x = torch.cat([x, pos_skip.clone()[:, :3]], dim=1)
        elif x_skip is None and not self.use_xyz:
            x = x

        x = self.nn(x)

        return x, pos_skip, batch_skip, point2curveidx_skip


class CurveSAModule(torch.nn.Module):

    def __init__(self, ratio, r, nn, curve_fps_arclen=None, use_curve_fps=False, global_nn=None, attend_nn=None, with_xyz=False, aggr_type='max', normalize_radius=False, **kwargs):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.curve_fps_arclen = curve_fps_arclen
        self.use_curve_fps = use_curve_fps
        self.with_xyz = with_xyz
        self.normalize_radius = r if normalize_radius else None
        self.conv = PointNetConv2(nn, add_self_loops=False, global_nn=global_nn, aggr_type=aggr_type, attend_nn=attend_nn, normalize_radius=self.normalize_radius)

    def forward(self, x, pos, batch, point2curveidx, **kwargs):
        # concatenate global xyz as feature
        if x is not None and self.with_xyz:
            x = torch.cat([x, pos.clone()[:, :3]], dim=1)
        elif x is None and self.with_xyz:
            x = pos.clone()[:, :3]

        # Run FPS to get subset of points
        if not self.use_curve_fps:
            idx = fps_pytorch3d(pos, batch, self.ratio)
        else:
            fps_module = CurveFPS(self.curve_fps_arclen)
            idx = fps_module(pos.clone(), batch.clone(), point2curveidx.clone())

        # run PointNet++ with geodesic "curve-groupings"
        row, col = radius_1d_group_subset(pos, idx, point2curveidx, batch, self.r)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)

        # update position and batch to downsampled
        pos, batch, point2curveidx = pos[idx], batch[idx], point2curveidx[idx]

        return x, pos, batch, point2curveidx, None, idx


class CurveFPModule(FPModule):
    def __init__(self, k, nn, with_xyz=False):
        super().__init__(k, nn, with_xyz)

    def forward(self, x, idx, x_skip, pos_skip, batch_skip, point2curveidx_skip=None, **kwargs):
        # Run KNN interpolation along the curve
        x = knn_interpolate_1D_pytorch3d(x, idx, pos_skip, batch_skip, point2curveidx_skip, k=self.k)  # (x, idx, pos_y, batch_y, point2curveidx_y, k):
        assert not torch.any(torch.isnan(x))

        # update interpolated feature
        if x_skip is not None and not self.with_xyz:
            x = torch.cat([x, x_skip.clone()], dim=1)
        elif x_skip is not None and self.with_xyz:
            x = torch.cat([x, x_skip.clone(), pos_skip.clone()[:, :3]], dim=1)
        elif x_skip is None and self.with_xyz:  # implies we are back to first layer
            x = torch.cat([x, pos_skip.clone()[:, :3]], dim=1)
        elif x_skip is None and not self.use_xyz:
            x = x

        x = self.nn(x)

        return x, pos_skip, batch_skip, point2curveidx_skip