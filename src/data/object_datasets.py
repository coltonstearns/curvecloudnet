import os.path as osp
import torch
from torch_geometric.data import (Data, Dataset)
import copy
import torch_geometric.transforms as T


class CurvesInMemoryDataset(Dataset):
    r"""
    Base class for any dataset that loads curve-level point data.
    """
    def __init__(self, datadir, npoints, resolution, line_density, laser_motion, split='train', dataset_source='', **kwargs):
        super().__init__(datadir, None, None, None)
        datapath = "data_%s_%s_%s_%s_%s.pth" % (npoints, resolution, line_density, laser_motion, split)
        self.data = torch.load(osp.join(datadir, datapath)).cpu()  # torch_geometric batch object
        self.split = split
        self.in_dim = 3
        self.use_additional_losses = kwargs['use_additional_losses']
        self.dataset_source = dataset_source

    def get(self, idx: int) -> Data:
        # contains data.pos, data.x, data.curvature, data.normals, data.y, and data.ids
        data = copy.copy(self.data[idx])
        data.pos = data.pos.float()
        data.normals = data.normals.float()
        data.curve_idxs = data.curve_idxs.long()

        # normalize-scale xyz coordinates
        data.pos = pc_normalize(data.pos)

        # apply standard training data augmentation
        if self.split == "train" and self.dataset_source == "shapenet-seg":
            transform = T.Compose([T.RandomScale((0.95, 1.05)),])
            pre_transform = T.NormalizeScale()
            data = transform(pre_transform(data))
            data.pos += (torch.rand((1, 3))-0.5) * 0.05

        return data

    def len(self) -> int:
        return torch.max(self.data.batch).item()


class SummerRoboticsDataset(CurvesInMemoryDataset):
    def __init__(self, datadir, npoints, dataset_source='', **kwargs):
        Dataset.__init__(self, datadir, None, None, None)
        datapath = "data_%s_%s.pth" % (npoints, "summer_robotics_test")
        self.data = torch.load(osp.join(datadir, datapath)).cpu()  # torch_geometric batch object
        self.split = 'val'
        self.in_dim = 3
        self.use_additional_losses = False
        self.dataset_source = dataset_source

    def get(self, idx: int) -> Data:
        return super().get(idx)

    def len(self) -> int:
        return super().len()


def pc_normalize(pc):
    centroid = torch.mean(pc, dim=0)
    pc = pc - centroid
    m = torch.max(torch.linalg.norm(pc, dim=1))
    pc = pc / m
    return pc