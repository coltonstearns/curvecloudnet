import torch.nn.functional as F
import torch
from torch import Tensor
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single
import torch.nn as nn
from typing import Optional
from src.models.utils.point_ops import curveidx_local2global, batch2ptr


class SymmetricCurve1DConvV2(nn.Module):

    def __init__(self, feat_dims=(64, 64, 128), kernel_size=5, bias=True, device=None, dtype=None, with_xyz=False, with_diff=False):
        """
        Args:
        """
        super(SymmetricCurve1DConvV2, self).__init__()
        self.kernel_size = kernel_size
        self.feat_dims = feat_dims
        self.with_xyz = with_xyz
        self.with_diff = with_diff
        kernel_size_unique = (kernel_size//2) + 1  # because symmetric
        conv_modules = []
        norm_modules = []
        for i in range(1, len(self.feat_dims)):
            input_dim = feat_dims[i-1] * 2 if (self.with_diff and i == 1) else feat_dims[i-1]
            conv_module = SymmetricConv1d(in_channels=input_dim, out_channels=feat_dims[i], kernel_size=kernel_size_unique,
                                          stride=1, padding="same", bias=bias, padding_mode='zeros', device=device, dtype=dtype)
            conv_modules.append(conv_module)
            norm_modules.append(nn.BatchNorm1d(feat_dims[i]))
        self.conv_modules = nn.ModuleList(conv_modules)
        self.norm_modules = nn.ModuleList(norm_modules)

    def forward(self, x, pos, batch, point2curveidx, **kwargs):
        """
        Forward pass of 1D convolution along each 3D polyline.
        """
        # prepare inputs
        device = pos.device
        point2curveidx_glob = curveidx_local2global(point2curveidx, batch)
        assert torch.all(point2curveidx_glob[1:] - point2curveidx_glob[:-1] >= 0)
        if x is None and self.with_xyz:
            x = pos
        elif x is not None and self.with_xyz:
            x = torch.cat([x, pos], dim=1)

        # estimate how much padding we need between curves
        nconvs = len(self.feat_dims) - 1
        padding = (self.kernel_size // 2) * nconvs
        ptr = batch2ptr(point2curveidx_glob, with_ends=True)  # **interior pointer idxs
        n_ptr = ptr.size(0)
        n_conv = x.size(0) + n_ptr*padding

        # identify which indices will added as padded
        if self.kernel_size > 1:  # when not just identity, add in padding
            pad_idxs = ptr.view(-1, 1) + torch.arange(n_ptr * (padding)).view(-1, padding).to(device)
            pad_idxs = pad_idxs.flatten()
        else:
            pad_idxs = torch.tensor([], device=x.device, dtype=torch.bool)
        valid_idxs = torch.ones(n_conv, device=x.device, dtype=torch.bool)
        valid_idxs[pad_idxs] = False

        # compute feature difference and add zero-padding in between curves
        if self.with_diff:
            x_diff = compute_feature_diffs(x, point2curveidx, batch)
            x = torch.cat([x, x_diff], dim=1)
        x_padded = torch.zeros((n_conv, x.size(1)), device=x.device)
        x_padded[valid_idxs] = x  # N' x feats

        # run 1d convolutions
        for i, conv in enumerate(self.conv_modules):
            x_padded = conv(x_padded.transpose(0, 1)).transpose(0, 1)
            x_padded = F.leaky_relu(self.norm_modules[i](x_padded))
        x = x_padded[valid_idxs]
        return x, pos, batch, point2curveidx


class SymmetricCurve1DConvFastV1(nn.Module):
    def __init__(self, feat_dims=(64, 64, 128), kernel_size=5, bias=True, device=None, dtype=None, with_xyz=False, with_diff=False):
        """
        Args:
        """
        super(SymmetricCurve1DConvFastV1, self).__init__()
        self.kernel_size = kernel_size
        self.feat_dims = feat_dims
        self.with_xyz = with_xyz
        self.with_diff = with_diff
        kernel_size_unique = (kernel_size//2) + 1  # because symmetric
        conv_modules = []
        norm_modules = []
        for i in range(1, len(self.feat_dims)):
            input_dim = feat_dims[i-1] * 2 if self.with_diff else feat_dims[i-1]
            conv_module = SymmetricConv1d(in_channels=input_dim, out_channels=feat_dims[i], kernel_size=kernel_size_unique,
                                          stride=1, padding="same", bias=bias, padding_mode='zeros', device=device, dtype=dtype)
            conv_modules.append(conv_module)
            norm_modules.append(nn.BatchNorm1d(feat_dims[i]))

        self.conv_modules = nn.ModuleList(conv_modules)
        self.norm_modules = nn.ModuleList(norm_modules)

    def forward(self, x, pos, batch, point2curveidx, **kwargs):
        """
        Forward pass of 1D convolution along each 3D polyline.
        """
        # prepare and verify inputs
        device = pos.device
        point2curveidx_glob = curveidx_local2global(point2curveidx, batch)
        assert torch.all(point2curveidx_glob[1:] - point2curveidx_glob[:-1] >= 0)
        if x is None and self.with_xyz:
            x = pos
        elif x is not None and self.with_xyz:
            x = torch.cat([x, pos], dim=1)

        # compute ptr form of batch
        ptr = batch2ptr(point2curveidx_glob)  # **interior pointer idxs
        n_ptr = ptr.size(0)
        n_conv = x.size(0) + n_ptr*(self.kernel_size//2)

        # compute padding to 1D conv.
        if self.kernel_size > 1:  # when not just identity, add in padding
            pad_idxs = ptr.view(-1, 1) + torch.arange(n_ptr * (self.kernel_size//2)).view(-1, self.kernel_size//2).to(device)
            pad_idxs = pad_idxs.flatten()
        else:
            pad_idxs = torch.tensor([], device=x.device, dtype=torch.bool)
        valid_idxs = torch.ones(n_conv, device=x.device, dtype=torch.bool)
        valid_idxs[pad_idxs] = False

        # apply 1D symmetric convolutions
        for i, conv in enumerate(self.conv_modules):
            # concatenate with curve gradients
            if self.with_diff:
                x_diff = compute_feature_diffs(x, point2curveidx, batch)
                x = torch.cat([x, x_diff], dim=1)

            # recompute padding
            x_padded = torch.zeros((n_conv, x.size(1)), device=x.device)
            x_padded[valid_idxs] = x  # N' x feats

            # apply convolution and activation
            x_padded = conv(x_padded.transpose(0, 1)).transpose(0, 1)
            x = x_padded[valid_idxs]
            norm = self.norm_modules[i]
            x = F.leaky_relu(norm(x))

        return x, pos, batch, point2curveidx


class SymmetricConv1d(_ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding: str = 'same',
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super(SymmetricConv1d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _single(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        # convert into symmetric weights and biases
        # Note: self.weight is tensor with size (out_channels, in_channels, stride)
        if weight.size(2) > 1:
            reflected = torch.flip(weight[:, :, 1:], dims=[2])
            weight = torch.cat([reflected, weight], dim=2)
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)


def compute_feature_diffs(x, point2curveidx, batch):
    # computer avg edge length of each curve
    point2curveidx_glob = curveidx_local2global(point2curveidx, batch)
    edges, edge_validity = x[1:] - x[:-1], (point2curveidx_glob[1:] - point2curveidx_glob[:-1]) == 0
    edges[~edge_validity] = 0

    # zero pad ends of edges and edge-validity
    zero_edge = torch.zeros((1, x.size(1))).to(x.device)
    edges = torch.cat([zero_edge, edges, zero_edge], dim=0)  # N+1 x feats
    edge_validity = torch.cat([torch.zeros(1, dtype=torch.bool).to(x.device), edge_validity, torch.zeros(1, dtype=torch.bool).to(x.device)])

    # take average of edges for per-pnt derivative (where applicable)
    edge_sums = edges[1:] + edges[:-1]  # N x feats
    edge_denom = torch.clip((edge_validity[1:].float() + edge_validity[:-1].float()).float(), min=1)
    per_pnt_diff = torch.abs(edge_sums / edge_denom[:, None])
    return per_pnt_diff

