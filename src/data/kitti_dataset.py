import torch
from torch_geometric.data import (Data, Dataset)
import yaml
import numpy as np
import cv2

from src.data.data_utils import absoluteFilePaths, polarmix
from src.visualization.mitsuba_render import render_pc_kitti

# Polarmix settings
# we apply polarmix to classes [car, bicycle, motorcycle, truck, other-vehicle, person, bicyclist, motorcyclist]
instance_classes = [1, 2, 3, 4, 5, 6, 7, 8]


class SemKITTI(Dataset):
    CURVE_THRESH = 0.08  # our heuristic for curve splicing on the KITTI dataset
    POS_NORMALIZE = 20

    def __init__(self, data_path, yaml_path, split='train', polarmix_aug=False):
        super().__init__(data_path, None, None, None)
        assert split in ["train", "val", "test"]
        self.split = split
        with open(yaml_path, 'r') as stream:
            self.semkittiyaml = yaml.safe_load(stream)
        self.polarmix_aug = polarmix_aug

        # load dataset info
        self.in_dim = 3 + 1  # xyz, reflectance
        self.learning_map = self.semkittiyaml['learning_map']
        self.split_dirs = self.semkittiyaml['split'][split]

        # load filepaths
        self.fpaths = []
        for i_folder in self.split_dirs:
            self.fpaths += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))

    def len(self):
        return len(self.fpaths)

    def get(self, idx: int, viz=False) -> Data:
        points, labels, reflectance, fpath = self._load_frame(idx)
        points, labels, reflectance = torch.from_numpy(points), torch.from_numpy(labels), torch.from_numpy(reflectance)[:, None]
        curve_idxs = self._get_curves(points)

        # polar mix augmentations
        if self.split == 'train' and self.polarmix_aug:
            points, reflectance, curve_idxs, labels = self._polarmix(points, reflectance, curve_idxs, labels)

        # if training, apply data augmentation
        if self.split == "train":
            points = torch.from_numpy(self._training_augmentation(points.data.numpy()))

        # when in debug mode of KITTI curves
        if viz:
            self._viz_kitti_curves(points, curve_idxs, idx)

        # normalize KITTI position to roughly lie within +-3
        points /= self.POS_NORMALIZE
        return Data(x=reflectance, pos=points, y=labels.flatten().long(), curve_idxs=curve_idxs.long(), fpath=fpath)

    def _load_frame(self, index):
        raw_data = np.fromfile(self.fpaths[index], dtype=np.float32).reshape((-1, 4))
        if self.split == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.fpaths[index].replace('velodyne', 'labels')[:-3] + 'label', dtype=np.int32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8), raw_data[:, 3], self.fpaths[index])

        return data_tuple

    def _get_curves(self, points):
        # KITTI dataset only gives us a single "sequential beam", so all points belong to the same 2D curve
        curve_idxs_2D = torch.zeros(points.size(0)).to(points.device)
        edge_splits_2D = (curve_idxs_2D[1:] - curve_idxs_2D[:-1]) != 0

        # compute each edge-length along a curve
        edges = points[1:].double() - points[:-1].double()
        edge_norms = torch.linalg.norm(edges, dim=-1)

        # as we move further away from (0, 0), we expect curves to be less dense --> correct for this
        radii = torch.linalg.norm(points[1:, :2], dim=-1)

        # identify where a curve is discontinuous
        curve_splits_3D = edge_norms > (self.CURVE_THRESH * np.sqrt(radii))
        curve_splits = curve_splits_3D | edge_splits_2D

        # Compute unique 3D curve index
        curve_idxs = torch.cumsum(curve_splits, dim=0)
        curve_idxs = torch.cat([torch.zeros(1), curve_idxs], dim=0)  # first point belongs to idx=0
        return curve_idxs

    def _apply_polarmix(self, points_orig, reflectance_orig, curve_idxs_orig, labels_orig):
        idx2 = np.random.randint(len(self.fpaths))
        pts2, labels2, reflectance2, fpath2 = self._load_frame(idx2)
        pts2, labels2, reflectance2 = torch.from_numpy(pts2), torch.from_numpy(labels2), torch.from_numpy(reflectance2)[:, None]
        curve_idxs2 = self._get_curves(pts2)
        curve_idxs2 += torch.max(curve_idxs_orig)
        inputs1 = torch.cat([points_orig, reflectance_orig, curve_idxs_orig.reshape(-1, 1)], dim=-1)
        inputs2 = torch.cat([pts2, reflectance2, curve_idxs2.reshape(-1, 1)], dim=-1)

        # polarmix inputs1 and inputs2
        alpha = (np.random.random() - 1) * np.pi
        beta = alpha + np.pi
        Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]  # x3
        inputs_aug, labels_aug = polarmix(inputs1.numpy(), labels_orig.numpy().flatten(), inputs2.numpy(),
                                          labels2.numpy().flatten(),
                                          alpha=alpha, beta=beta,
                                          instance_classes=instance_classes,
                                          Omega=Omega)

        # extract updated points, reflectance, labels, and curve idxs
        points = inputs_aug[:, :3]
        reflectance = inputs_aug[:, 3:4]
        curve_idxs = inputs_aug[:, 4]
        points, labels, reflectance = torch.from_numpy(points), torch.from_numpy(labels_aug).view(-1, 1), torch.from_numpy(reflectance)
        curve_idxs = torch.from_numpy(curve_idxs)
        _, curve_idxs = torch.unique(curve_idxs, return_inverse=True)  # unify and densify curve indices

        return points, reflectance, curve_idxs, labels

    def _training_augmentation(self, xyz):
        # rotation augmentation
        rotate_rad = np.deg2rad(np.random.random() * 180) - np.pi / 2
        c, s = np.cos(rotate_rad), np.sin(rotate_rad)
        j = np.matrix([[c, s], [-s, c]])
        xyz[:, :2] = np.dot(xyz[:, :2], j)

        # flip augmentation: flip x , y or x+y
        flip_type = np.random.choice(4, 1)
        if flip_type == 1:
            xyz[:, 0] = -xyz[:, 0]
        elif flip_type == 2:
            xyz[:, 1] = -xyz[:, 1]
        elif flip_type == 3:
            xyz[:, :2] = -xyz[:, :2]

        # scale augmentation
        noise_scale = np.random.uniform(0.95, 1.05)
        xyz[:, 0] = noise_scale * xyz[:, 0]
        xyz[:, 1] = noise_scale * xyz[:, 1]

        # random pertubation augmentation
        trans_std = [0.1, 0.1, 0.1]
        noise_translate = np.array([np.random.normal(0, trans_std[0], 1),
                                    np.random.normal(0, trans_std[1], 1),
                                    np.random.normal(0, trans_std[2], 1)]).T

        xyz[:, 0:3] += noise_translate

        return xyz

    def _viz_kitti_curves(self, pc, curve_idxs, idx, suffix=""):
        import matplotlib.pyplot as plt
        print("In debug mode... Rendering KITTI scene.")
        pc, curve_idxs = pc.data.cpu().numpy(), curve_idxs.data.cpu().numpy()

        curve_reds = [float(hash(str(idx) + 'r') % 256) / 255 for idx in curve_idxs.tolist()]
        curve_greens = [float(hash(str(idx) + 'g') % 256) / 255 for idx in curve_idxs.tolist()]
        curve_blues = [float(hash(str(idx) + 'b') % 256) / 255 for idx in curve_idxs.tolist()]
        clrs = np.stack([curve_reds, curve_greens, curve_blues], axis=1)

        # render pc
        img = render_pc_kitti(pc, clrs, point_radius=0.0009)
        img = np.array(img)
        cv2.imwrite("./kitti_%s%s.png" % (idx, suffix), img *255)
        # cv2.imwrite("./kitti_ego_%s%s.png" % (idx, suffix), img_ego *255)


# for debugging
if __name__ == '__main__':
    data_path = "/home/colton/Documents/kitti/sequences"
    yaml_path = "/home/colton/Documents/curvenet/configs/semantic-kitti.yaml"
    train_dataset = SemKITTI(data_path, yaml_path, split="train")
    train_dataset.get(8, viz=True)
