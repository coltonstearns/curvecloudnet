import pickle
import torch
from torch_geometric.data import (Data, Dataset)
import yaml
import numpy as np
import os
import cv2

from src.visualization.mitsuba_render import render_pc_kitti
from src.data.data_utils import polarmix

# polarmix instance classes
INSTANCE_CLASSES = [2, 3, 4, 5, 6, 7, 9, 10]


class SemNuScenes(Dataset):
    CURVE_THRESH = 0.08  # a good final curve thresh
    POS_NORMALIZE = 20

    def __init__(self, data_path, yaml_path, nusc, split="train", polarmix_aug=False):
        super().__init__(data_path, None, None, None)
        with open(yaml_path, 'r') as stream:
            self.nuscyaml = yaml.safe_load(stream)
        self.in_dim = 3 + 1  # xyz, reflectance
        self.polarmix_aug = polarmix_aug

        # load nuscenes dataset info
        self.learning_map = self.nuscyaml['learning_map']
        splitfile = os.path.join(data_path, 'nuscenes_infos_%s.pkl' % split)
        self.split = split
        with open(splitfile, 'rb') as f:
            data = pickle.load(f)
        self.nusc_infos = data['infos']
        self.data_path = data_path

        # initialize nuscenes dataset
        self.nusc = nusc

    def len(self):
        return len(self.nusc_infos)

    def get(self, idx: int, viz: bool = False) -> Data:
        points, labels, reflectance, beam_ids, seg_fname = self._load_frame(idx)
        points, labels, reflectance, beam_ids = torch.from_numpy(points), torch.from_numpy(labels), torch.from_numpy(reflectance)[:, None], torch.from_numpy(beam_ids)

        # compute 3D curve indices
        points, curve_idxs, labels, reflectance, inv_reorder = self._get_curves(points, beam_ids, labels, reflectance)

        # polar mix augmentations
        if self.split == 'train' and self.polarmix_aug:
            points, reflectance, curve_idxs, labels = self._apply_polarmix(points, reflectance, curve_idxs, labels)

        # apply training augmentation
        if self.split == "train":
            points = torch.from_numpy(self._training_augmentation(points.data.numpy()))

        # when in debug mode of KITTI curves
        if viz:
            self._viz_nusc_curves(points, curve_idxs, idx)

        # normalize positions to roughly lie in +-3
        points /= self.POS_NORMALIZE
        return Data(x=reflectance, pos=points, y=labels.flatten().long(), curve_idxs=curve_idxs.long(), fname=seg_fname, reorder=inv_reorder)

    def _load_frame(self, index):
        # find lidar file name
        info = self.nusc_infos[index]
        lidar_filename = os.path.join(self.data_path, "/".join(info['lidar_path'].split("/")[-3:]))
        while not os.path.exists(lidar_filename):
            print("No filename in dataset: %s. Using other random sample instead..." % lidar_filename)
            index = (index + 1) % len(self)
            info = self.nusc_infos[index]
            lidar_filename = os.path.join(self.data_path, "/".join(info['lidar_path'].split("/")[-3:]))
        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']

        # load lidar points
        points = np.fromfile(lidar_filename, dtype=np.float32, count=-1).reshape([-1, 5])
        if self.split != "test":
            lidarseg_labels_filename = os.path.join(self.nusc.dataroot, self.nusc.get('lidarseg', lidar_sd_token)['filename'])
            seg_filename = self.nusc.get('lidarseg', lidar_sd_token)['filename'].split("/")[-1]
            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        else:
            points_label = np.ones((points.shape[0], 1), dtype=np.uint8)  # testing split --> create dummy labels
            seg_filename = lidar_sd_token + "_lidarseg.bin"

        # load points, reflectance, and beam id
        data_tuple = (points[:, :3], points_label.astype(np.uint8), points[:, 3], points[:, 4], seg_filename)
        return data_tuple

    def _get_curves(self, points, beam_idxs, labels, reflectance):
        # sort points by beam index
        reorder = torch.sort(beam_idxs, stable=True)[1]  # ordering is based on sensor acquisition
        inv_reorder = torch.empty_like(reorder)
        inv_reorder[reorder] = torch.arange(points.size(0)).to(reorder)
        points = points[reorder]
        beam_idxs = beam_idxs[reorder]
        labels = labels[reorder]
        reflectance = reflectance[reorder]

        # define new curves when we move from one laser beam to the next
        edge_splits_2D = (beam_idxs[1:] - beam_idxs[:-1]) != 0

        # compute each edge-length along a curve
        edges = points[1:].double() - points[:-1].double()
        edge_norms = torch.linalg.norm(edges, dim=-1)

        # as we move further away from (0, 0), we expect curves to be less dense --> correct for this
        radii = torch.linalg.norm(points[1:, :2], dim=-1)

        # identify where a curve is discontinuous
        curve_splits_3D = edge_norms > (self.CURVE_THRESH * torch.sqrt(radii))
        curve_splits = curve_splits_3D | edge_splits_2D

        # Compute unique 3D curve index
        curve_idxs = torch.cumsum(curve_splits, dim=0)
        curve_idxs = torch.cat([torch.zeros(1).to(curve_idxs), curve_idxs], dim=0)  # first point belongs to idx=0
        return points, curve_idxs, labels, reflectance, inv_reorder

    def _training_augmentation(self, xyz):
        # rotation augmentation
        rotate_rad = np.deg2rad(np.random.random() * 360) - np.pi
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

    def _apply_polarmix(self, points_orig, reflectance_orig, curve_idxs_orig, labels_orig):
        idx2 = np.random.randint(len(self.nusc_infos))
        pts2, labels2, reflectance2, beam_ids2, fpath2 = self._load_nusc_data(idx2)
        pts2, labels2, reflectance2, beam_ids2 = torch.from_numpy(pts2), torch.from_numpy(labels2), torch.from_numpy(reflectance2)[:, None], torch.from_numpy(beam_ids2)
        pts2, curve_idxs2, labels2, reflectance2, inv_reorder2 = self._get_curves(pts2, beam_ids2, labels2, reflectance2)
        curve_idxs2 += torch.max(curve_idxs_orig)
        inputs1 = torch.cat([points_orig, reflectance_orig, curve_idxs_orig.reshape(-1, 1)], dim=-1)
        inputs2 = torch.cat([pts2, reflectance2, curve_idxs2.reshape(-1, 1)], dim=-1)
        # polarmix
        alpha = (np.random.random() - 1) * np.pi
        beta = alpha + np.pi
        Omega = [np.random.random() * np.pi * 2 / 3, (np.random.random() + 1) * np.pi * 2 / 3]  # x3
        inputs_aug, labels_aug = polarmix(inputs1.numpy(), labels_orig.numpy().flatten(), inputs2.numpy(),
                                          labels2.numpy().flatten(),
                                          alpha=alpha, beta=beta,
                                          instance_classes=INSTANCE_CLASSES,
                                          Omega=Omega)

        # extract updated points, reflectance, labels, and curve idxs
        points = inputs_aug[:, :3]
        reflectance = inputs_aug[:, 3:4]
        curve_idxs = inputs_aug[:, 4]
        points, labels, reflectance = torch.from_numpy(points), torch.from_numpy(labels_aug).view(-1, 1), torch.from_numpy(reflectance)
        curve_idxs = torch.from_numpy(curve_idxs)
        _, curve_idxs = torch.unique(curve_idxs, return_inverse=True)  # unify and densify curve indices
        return points, reflectance, curve_idxs, labels

    def _viz_nusc_curves(self, pc, curve_idxs, idx, suffix=""):
        print("In debug mode... Rendering nuScenes scan.")
        pc, curve_idxs = pc.data.cpu().numpy(), curve_idxs.data.cpu().numpy()
        curve_reds = [float(hash(str(idx) + 'r') % 256) / 255 for idx in curve_idxs.tolist()]
        curve_greens = [float(hash(str(idx) + 'g') % 256) / 255 for idx in curve_idxs.tolist()]
        curve_blues = [float(hash(str(idx) + 'b') % 256) / 255 for idx in curve_idxs.tolist()]
        clrs = np.stack([curve_reds, curve_greens, curve_blues], axis=1)

        # render pc
        pc = pc + np.array([0, 0, 20])
        img = render_pc_kitti(pc, clrs, point_radius=0.0025)
        img = np.array(img ** (1.0 / 2.2))
        cv2.imwrite("./nusc_%s%s.png" % (idx, suffix), img *255)


# for debugging
if __name__ == '__main__':
    from nuscenes import NuScenes
    nusc_data_root = "/media/colton/ColtonSSD/nuscenes-raw"
    nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_data_root, verbose=True)
    yaml_path = "/home/colton/Documents/curvenet/configs/nuscenes.yaml"
    splitfile = "/home/colton/Documents/curvenet/nuscenes_infos_val.pkl"
    train_dataset = SemNuScenes(nusc_data_root, yaml_path, nusc, split="val")
    for i in range(1000):
        train_dataset.get(i)

