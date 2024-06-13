import torch
import torch.nn as nn
from src.models.utils.point_ops import curveidx_local2global, batch2ptr
from torch_scatter import scatter_min


class CurveFPS(nn.Module):

    def __init__(self, arclen_spacing=0.3):
        """
        arclen_spacing (float): sample a point for each <arclen_spacing> units along a curve
        """
        super().__init__()
        self.arclen_spacing = arclen_spacing

    def forward(self, pos, batch, point2curveidx):
        # compute edge lengths
        point2curveidx_glob = curveidx_local2global(point2curveidx, batch)
        edges, edge_validity = pos[1:] - pos[:-1], (point2curveidx_glob[1:] - point2curveidx_glob[:-1]) == 0
        edge_norms = torch.linalg.norm(edges, dim=-1)
        edge_norms[~edge_validity] = 0

        # compute geodesic functions
        ptr_glob = torch.cat([torch.tensor([0]).to(pos.device), batch2ptr(point2curveidx_glob)])
        point2curvestartidx_glob = ptr_glob[point2curveidx_glob]
        geodesic_lengths = torch.cat([torch.tensor([0]).to(pos.device), torch.cumsum(edge_norms, dim=0)])
        geodesic_lengths -= geodesic_lengths[point2curvestartidx_glob]

        # get evenly-spaced samplings
        geodesic_lengths += ((point2curvestartidx_glob * 117 * torch.rand(1).to(pos.device)) % self.arclen_spacing)  # add random offsets
        geodesic_idxs = torch.round(geodesic_lengths / self.arclen_spacing)
        interval_starts = torch.cat([torch.ones(1, dtype=torch.bool, device=pos.device), (geodesic_idxs[1:] - geodesic_idxs[:-1]) != 0])
        interval_starts[point2curvestartidx_glob] = True

        # get indices
        selection_point_idxs = torch.where(interval_starts)[0]

        # select idxs
        return selection_point_idxs.long()


class VoxelFPS(nn.Module):

    def __init__(self, voxel_size=0.05):
        """
        arclen_spacing (float): sample a point for each <arclen_spacing> units along a curve
        """
        super().__init__()
        self.voxel_size = voxel_size

    def forward(self, pos, batch):
        voxels = torch.floor(pos / self.voxel_size).long()
        voxels = torch.cat([batch.view(-1, 1), voxels], dim=-1)
        voxels_u, voxel_idxs = torch.unique(voxels, dim=0, return_inverse=True)  #
        vox_dists = torch.linalg.norm(voxels[:, 1:] - (pos / self.voxel_size), dim=-1)  # array of float distances
        vox_dists += torch.rand(vox_dists.size()).to(vox_dists) * self.voxel_size/4  # randomness, but loosely centered within voxel

        size = int(voxel_idxs.max().item() + 1)
        _, min_voxel_idxs = scatter_min(vox_dists.view(-1, 1), voxel_idxs, dim=0, out=None, dim_size=size)
        return min_voxel_idxs.flatten()



