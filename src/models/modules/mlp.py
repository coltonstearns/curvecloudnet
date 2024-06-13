import torch
from torch_geometric.nn import MLP


class SharedMLP(torch.nn.Module):

    def __init__(self, dims, use_bias=False, with_xyz=False, act='leaky_relu', **kwargs):
        super().__init__()
        plain_last = kwargs.get("plain_last", True)
        norm = kwargs.get("norm", "batch_norm")
        dropout = kwargs.get("dropout", 0.0)
        mlp_func = MLP
        self.mlp = mlp_func(dims, dropout=dropout, norm=norm, plain_last=plain_last, act=act, bias=use_bias)
        self.with_xyz = with_xyz

    def forward(self, x, pos, batch, point2curveidx=None, **kwargs):
        if x is None and self.with_xyz:
            x = pos
        elif x is not None and self.with_xyz:
            x = torch.cat([x, pos], dim=1)
        x = self.mlp(x)
        return x, pos, batch, point2curveidx