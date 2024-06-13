from typing import Callable, Optional, Union, Any
import torch
from torch import Tensor
from torch_geometric.nn.conv.point_conv import PointNetConv
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor
from torch_scatter import scatter_max, scatter_add, scatter_mean
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_sparse import SparseTensor, set_diag
import torch.nn.functional as F


class PointNetConv2(PointNetConv):
    """
    Our modification of PointConv from PytorchGeometric.
    Implements attention pooling, as well as radius normalization from PointNext.
    """

    def __init__(self, local_nn: Optional[Callable] = None,
                 global_nn: Optional[Callable] = None,
                 attend_nn: Optional[Callable] = None,
                 add_self_loops: bool = True, aggr_type='max', normalize_radius=None, **kwargs):
        assert aggr_type in ['max', 'attend', 'mean', 'weighted-sum']
        kwargs.setdefault('aggr', 'max' if aggr_type == 'max' else 'sum')
        super().__init__(**kwargs)

        self.local_nn = local_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops
        self.attend_nn = attend_nn
        self.aggr_type = aggr_type
        self.normalize_radius = normalize_radius

        self.reset_parameters()

    def forward(self, x: Union[OptTensor, PairOptTensor],
                pos: Union[Tensor, PairTensor], edge_index: Adj) -> (Any, Any):

        if not isinstance(x, tuple):
            x: PairOptTensor = (x, None)

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(
                    edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: PairOptTensor, pos: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, size=None)

        if self.global_nn is not None:
            out = self.global_nn(out)

        return out

    def message(self, x_j: Optional[Tensor], pos_i: Tensor,
                pos_j: Tensor) -> Tensor:
        msg = pos_j - pos_i
        if self.normalize_radius is not None:
            msg /= self.normalize_radius
        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

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
        elif self.aggr_type == 'mean':
            return scatter_mean(inputs, index, dim=self.node_dim, out=None, dim_size=dim_size)
        elif self.aggr_type == "weighted-sum":
            inputs_attend = self.attend_nn(inputs)
            weights = F.sigmoid(inputs_attend)
            out_feats = scatter_add(inputs*weights, index, dim=self.node_dim)
            return out_feats
        else:  # self.aggr_type == 'attend':
            inputs_attend = self.attend_nn(inputs)
            weights = softmax(inputs_attend, index, dim=self.node_dim)
            out_feats = scatter_add(inputs*weights, index, dim=self.node_dim)
            return out_feats


