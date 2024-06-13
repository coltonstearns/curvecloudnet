import torch
import torch.nn as nn
from typing import Callable, Optional, Union
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor
from src.models.utils.point_ops import knn_ball_group_pytorch3d, to_batch_padded
from torch_geometric.nn.inits import reset
from torch_scatter import scatter_max, scatter_add
from torch_geometric.utils import softmax
from torch_cluster import knn
import frnn
import torch.nn.functional as F


class DynamicEdgeConv(MessagePassing):
    r"""The dynamic edge convolutional operator from the `"Dynamic Graph CNN
    for Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper
    (see :class:`torch_geometric.nn.conv.EdgeConv`), where the graph is
    dynamically constructed using nearest neighbors in the feature space.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            `:obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.* defined by :class:`torch.nn.Sequential`.
        k (int): Number of nearest neighbors.
        aggr (string): The aggregation operator to use (:obj:`"add"`,
            :obj:`"mean"`, :obj:`"max"`). (default: :obj:`"max"`)
        num_workers (int): Number of workers to use for k-NN computation.
            Has no effect in case :obj:`batch` is not :obj:`None`, or the input
            lies on the GPU. (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V}|, F_{in}), (|\mathcal{V}|, F_{in}))`
          if bipartite,
          batch vector :math:`(|\mathcal{V}|)` or
          :math:`((|\mathcal{V}|), (|\mathcal{V}|))`
          if bipartite *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn: Callable, k: int = None, radius: int = None, aggr: str = 'max',
                 num_workers: int = 1, use_knn=True, **kwargs):
        super().__init__(aggr=aggr, flow='source_to_target', **kwargs)

        if knn is None:
            raise ImportError('`DynamicEdgeConv` requires `torch-cluster`.')

        self.nn = nn
        self.r = radius
        self.k = k
        self.num_workers = num_workers
        self.use_knn = use_knn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(
            self, x: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        if x[0].dim() != 2:
            raise ValueError("Static graphs not supported in DynamicEdgeConv")

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        if self.use_knn:
            row, col = knn_ball_group_pytorch3d(x[1], x[0], b[1], b[0], operation="knn", knn=self.k)
        else:
            row, col = knn_ball_group_pytorch3d(x[1], x[0], b[1], b[0], operation="ball-group", radius=self.r)
        edge_index = torch.stack([row, col], dim=0)
        edge_index = edge_index.flip([0])

        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn}, k={self.k})'


class DGCNNLayer(DynamicEdgeConv):

    def __init__(self, nn: Callable, k: int, aggr: str = 'max', num_workers: int = 1, with_xyz=False, **kwargs):
        self.with_xyz = with_xyz
        super(DGCNNLayer, self).__init__(nn, k, None, aggr, num_workers, **kwargs)

    def forward(self, x, pos, batch, point2curveidx=None, **kwargs):
        if x is None and self.with_xyz:
            x = pos
        elif x is not None and self.with_xyz:
            x = torch.cat([x, pos], dim=1)
        out = super(DGCNNLayer, self).forward(x, batch)

        return out, pos, batch, point2curveidx


class DGCNNLayerRadius(DynamicEdgeConv):

    def __init__(self, nn: Callable, r: int, aggr: str = 'max', num_workers: int = 1, with_xyz=False, **kwargs):
        self.with_xyz = with_xyz
        super(DGCNNLayerRadius, self).__init__(nn, None, r, aggr, num_workers, use_knn=False, **kwargs)

    def forward(self, x, pos, batch, point2curveidx=None, **kwargs):
        if x is None and self.with_xyz:
            x = pos
        elif x is not None and self.with_xyz:
            x = torch.cat([x, pos], dim=1)
        out = super(DGCNNLayerRadius, self).forward(x, batch)

        return out, pos, batch, point2curveidx


class StaticEdgeConv(MessagePassing):
    def __init__(self, nn: Callable, k: int, aggr: str = 'max',
                 num_workers: int = 1,
                 attend_nn: Optional[Callable] = None,
                 aggr_type = 'max',
                 r=1.0,
                 use_fast_knn=True,
                 use_sparse_feat_agg=False,
                 **kwargs):
        super().__init__(aggr=aggr, flow='source_to_target', **kwargs)

        self.nn = nn
        self.k = k
        self.r = r
        self.num_workers = num_workers
        self.attend_nn = attend_nn
        self.aggr_type = aggr_type
        self.use_fast_knn = use_fast_knn
        self.use_sparse_feat_agg = use_sparse_feat_agg
        assert aggr_type in ['max', 'attend', 'mean', 'weighted-sum']

    def forward(self, x: Union[Tensor, PairTensor], pos: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None) -> Tensor:
        if self.use_sparse_feat_agg:
            return self.forward_slow(x, pos, batch)
        else:
            return self.forward_fast(x, pos, batch)

    def forward_fast(
            self, x: Union[Tensor, PairTensor], pos: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None) -> Tensor:
        """"""
        # get KNN --> keep in dense batch form because faster (since we have fixed number of neighbors)
        idxs, lengths_p2, mask_p1 = knn_ball_group_pytorch3d(pos, pos, batch, batch, operation="knn", knn=self.k, radius=self.r, return_dense=True)

        # add self loops
        B, N = idxs.size(0), idxs.size(1)
        self_loops = torch.arange(N).reshape(1, N, 1).repeat(B, 1, 1).to(idxs)
        idxs = torch.cat([self_loops, idxs], dim=2)  # now is (B x N x K+1)

        # gather features --> size (B, N_max, C)
        x = to_batch_padded(x, batch)[0]
        feats = frnn.frnn_gather(x, idxs, lengths_p2)
        feats = torch.cat([feats, feats[:, :, 0:1, :] - feats], dim=-1)
        feats = feats.view(-1, feats.size(-1))

        # process batch of features
        feats = self.nn(feats)
        feats = feats.view(B, N, self.k+1, feats.size(-1))

        # pool features
        mask = ((idxs != -1) * mask_p1.unsqueeze(-1)).bool()
        if self.aggr_type == "mean":
            feats[~mask] = 0
            feats_sum = torch.sum(feats, dim=2)
            counts = torch.sum(mask, dim=2)
            feats = feats_sum / counts.unsqueeze(-1)
        elif self.aggr_type == "max":
            feats[~mask] = -1e2
            feats = torch.max(feats, dim=2)[0]
        elif self.aggr_type == "weighted-sum":
            inputs_attend = self.attend_nn(feats.view(B*N*(self.k+1), -1))
            inputs_attend = inputs_attend.view(B, N, self.k+1, -1)
            weights = F.sigmoid(inputs_attend)
            weights[~mask] = 0
            totals = torch.sum(weights, dim=2, keepdim=True)
            weights /= torch.clamp(totals, min=1e-3)
            feats = torch.sum(feats*weights, dim=2)
        elif self.aggr_type == "attend":
            inputs_attend = self.attend_nn(feats.view(B*N*(self.k+1), -1))
            inputs_attend = inputs_attend.view(B, N, self.k+1, -1)
            inputs_attend[~mask] = -5e2
            weights = F.softmax(inputs_attend, dim=2)
            feats = torch.sum(feats*weights, dim=2)

        # feats is currently (B x N_max x C). We need BN x C
        feats = feats[mask_p1]
        return feats

    def forward_slow(self, x: Union[Tensor, PairTensor], pos: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None) -> Tensor:
        # set up pos and batch for GNN processing
        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)
        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        # apply GNN message passing
        row, col = knn_ball_group_pytorch3d(pos[1], pos[0], b[1], b[0], operation="knn", knn=self.k, radius=self.r, return_dense=False, accel_knn=self.use_fast_knn)
        edge_index = torch.stack([row, col], dim=0)
        edge_index = edge_index.flip([0])
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> (Tensor, Tensor):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.
        """
        if self.aggr_type == 'max':
            return scatter_max(inputs, index, dim=self.node_dim, out=None, dim_size=dim_size)[0]
        else:  # self.aggr_type == 'attend':
            inputs_attend = self.attend_nn(inputs)
            weights = softmax(inputs_attend, index, dim=self.node_dim)
            out_feats = scatter_add(inputs * weights, index, dim=self.node_dim)

        return out_feats

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn}, k={self.k})'


class SGCNNLayer(StaticEdgeConv):

    def __init__(self, nn: Callable, k: int, aggr: str = 'max', r: float = 1.0, num_workers: int = 1, with_xyz=False,
                 attend_nn: Optional[Callable] = None, aggr_type='max', use_sparse_feat_agg=False, **kwargs):
        self.with_xyz = with_xyz
        super(SGCNNLayer, self).__init__(nn, k, aggr, num_workers, attend_nn, aggr_type, r=r, use_sparse_feat_agg=use_sparse_feat_agg, **kwargs)

    def forward(self, x, pos, batch, point2curveidx=None, **kwargs):
        if x is None and self.with_xyz:
            x = pos
        elif x is not None and self.with_xyz:
            x = torch.cat([x, pos], dim=1)
        out = super(SGCNNLayer, self).forward(x, pos, batch)

        return out, pos, batch, point2curveidx
