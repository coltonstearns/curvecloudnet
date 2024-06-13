import torch
import torch.nn as nn
from typing import Callable, Optional, Union


class SkipConnect(nn.Module):
    def __init__(self, nn: Callable, num_skips=1):
        super(SkipConnect, self).__init__()
        self.num_skips = num_skips
        self.nn = nn

    def forward(self, xs, pos, batch, point2curveidx=None, **kwargs):
        x = torch.cat(xs, dim=1)
        out = self.nn(x)
        return out, pos, batch, point2curveidx