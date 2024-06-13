import os.path as osp
import torch
from torch_geometric.data import (Data, Dataset)
import yaml
import numpy as np
import cv2
import json
import glob
from os.path import join
import wandb

from src.visualization.mitsuba_render import render_pc_audi
from torch_cluster import knn
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp


class SemA2D2(Dataset):
    CURVE_THRESHES = [0.1, 0.17, 0.1, 0.12, 0.1]
    KNN = [4, 4, 4, 3, 4]
    POS_NORMALIZE = 30
    
    def __init__(self, data_path, yaml_path, split):
        super().__init__()
        self.in_dim = 3 + 1  # xyz and reflectance
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.split = split
        assert split in ['train', 'val', 'test']
        self.root = osp.join(data_path, "Segmentation", "semantic_%s" % self.split)
        self.id2name = self.config['labels']
        self.id2color = self.config['color_map']
        self.Hashed2label = self.config['Hashed2label'] 
        self.learning_map = self.config['learning_map']
        self.learning_map_inv = self.config['learning_map_inv']
        self.learning_ignore = self.config['learning_ignore']

        # load sensor config
        self.sensor_config_fpath = self.config['sensor_configuration_file']
        with open(self.sensor_config_fpath, 'r') as f:
            self.sensor_config = json.load(f)

        # get lidar file folders
        self.lidar_fnames = sorted(glob.glob(join(self.root, '*/lidar/cam_front_center/*.npz')))

        # remove files with missing labels in training set
        self.delete_path_list = self.config['missing_path']
        for i in range(len(self.delete_path_list)):
            path = osp.join(data_path, self.delete_path_list[i])
            if path in self.lidar_fnames:
                self.lidar_fnames.remove(path)

    # from lidar + .npz to label + .png
    def lidarfname_to_labelfname(self, file_name_lidar):
        file_name_image = file_name_lidar.split('/')
        file_name_image = file_name_image[-1].split('.')[0]
        file_name_image = file_name_image.split('_')
        file_name_image = file_name_image[0] + '_' + 'label_' + file_name_image[2] + '_' + file_name_image[3] + '.png'
        return file_name_image
    
    # fix image_distortion
    def undistort_frontcenter_image(self, image):
        """
        Assumes we're only operating on the front_center camera (for which we have labels)
        """
        # get distortion parameters from config file
        intr_mat_undist = np.asarray(self.sensor_config['cameras']['front_center']['CamMatrix'])
        intr_mat_dist = np.asarray(self.sensor_config['cameras']['front_center']['CamMatrixOriginal'])
        dist_parms = np.asarray(self.sensor_config['cameras']['front_center']['Distortion'])

        # undistort label-image
        h, w = image.shape[:2]
        mapx, mapy = cv2.initUndistortRectifyMap(intr_mat_dist, dist_parms, None, intr_mat_undist, (w, h), 5)
        dst = cv2.remap(image, mapx, mapy, cv2.INTER_NEAREST)
        return dst
    
    def len(self):
        """Get total number of samples"""
        return len(self.lidar_fnames)

    # train points augment -- copied from dataset.py
    def _training_augmentation(self, xyz):
        # flip augmentation: flip y axis on audi
        flip_type = np.random.choice(2, 1)
        if flip_type == 1:
            xyz[:, 1] = -xyz[:, 1]

        # scale augmentation
        noise_scale = np.random.uniform(0.95, 1.05)
        xyz[:, 0] = noise_scale * xyz[:, 0]
        xyz[:, 1] = noise_scale * xyz[:, 1]

        return xyz

    def _load_frame(self, idx):
        # load lidar data
        lidar_data_path = self.lidar_fnames[idx]
        lidar_data = np.load(lidar_data_path)

        # load label image
        seq_name = lidar_data_path.split('/')[-4]
        label_image_name = self.lidarfname_to_labelfname(lidar_data_path)
        camera_name = lidar_data_path.split('/')[-2]
        label_image_path = join(self.root, seq_name, 'label/', camera_name, label_image_name)
        label_image = cv2.imread(label_image_path)

        # undistort label image s.t. it matches our LiDAR projections
        label_image = self.undistort_frontcenter_image(label_image)

        # compute LiDAR per-point labels
        rows = (lidar_data['row'] + 0.5).astype(np.int64)
        cols = (lidar_data['col'] + 0.5).astype(np.int64)
        colors = label_image[rows, cols, :]

        # get labels
        label = []
        for i in range(lidar_data['points'].shape[0]):
            number = colors[i][0] * 1 + colors[i][1] * 100 + colors[i][2] * 10000
            label.append(self.learning_map[self.Hashed2label[number]])
        label = np.array(label)

        return lidar_data, label

    # get data piece from dataset
    def get(self, idx: int, viz=False) -> Data:
        # load lidar point cloud as well as corresponding labels
        lidar_dict, label = self._load_frame(idx)

        # compute curve-idxs based on heuristics (Audi doesn't give LIDAR beam-IDs)
        curve_idxs, points, label, reflectance = self._get_curves(lidar_dict, label)

        # optionally visualize
        if viz:
            self._viz_audi_curves(points, curve_idxs.detach(), idx)

        # normalize positions to roughly lie within +-3; format reflectance
        points = torch.from_numpy(points)
        points /= self.POS_NORMALIZE
        reflectance = torch.from_numpy(reflectance).view(-1, 1) / 255

        # if training, apply data augmentation
        if self.split == "train":
            points = torch.from_numpy(self._training_augmentation(points.numpy())).float()
        return Data(x=reflectance, pos=points.float(), y=torch.from_numpy(label).long(), curve_idxs=curve_idxs.long())

    def _viz_audi_curves(self, pc, curve_idxs, idx):
        print("Visualizing... Rendering Audi scene.")
        curve_reds = [float(hash(str(idx) + 'r') % 256) / 255 for idx in curve_idxs.tolist()]
        curve_greens = [float(hash(str(idx) + 'g') % 256) / 255 for idx in curve_idxs.tolist()]
        curve_blues = [float(hash(str(idx) + 'b') % 256) / 255 for idx in curve_idxs.tolist()]
        clrs = np.stack([curve_reds, curve_greens, curve_blues], axis=1)
        img = render_pc_audi(pc, clrs, point_radius=0.0025)
        img = np.array(img)
        cv2.imwrite("./audi_%s.png" % idx, img * 255)

    def _get_curves(self, lidar_data, label):
        """
        Perform a KNN-connectivity approximation to compute directly 3D curves.
        We perform this because A2D2 does NOT provide point aquisition timesteps
        """
        # iterate through each lidar
        lidar_ids = np.unique(lidar_data['lidar_id'].astype(np.int64))
        per_sensor_pts, per_sensor_labels, per_sensor_curveidxs, per_sensor_refs, total_curves = [], [], [], [], 0
        for id in lidar_ids:
            # get this specific LiDAR's sensor readings and labels
            this_pnts = lidar_data['points'][lidar_data['lidar_id'] == id]
            this_reflectances = lidar_data['reflectance'][lidar_data['lidar_id'] == id]
            this_labs = label[lidar_data['lidar_id'] == id]
            this_tsteps = lidar_data['timestamp'][lidar_data['lidar_id'] == id].reshape(-1, 1)

            # run KNN to get approximate connectivity graph
            this_pnts_trch = torch.from_numpy(this_pnts)
            batch = torch.zeros(this_pnts_trch.size(0))
            k = self.KNN[id]
            assign_index = knn(this_pnts_trch, this_pnts_trch, k=k, batch_x=batch, batch_y=batch)

            # filter KNN by neighbor distances - only allow distances lower than specified CURVE_THRESHOLD
            edge_lens = torch.norm(this_pnts_trch[assign_index[0]] - this_pnts_trch[assign_index[1]], dim=1)
            sqrt_radii = torch.sqrt(torch.linalg.norm(this_pnts_trch[:, :2], dim=-1))
            edge_len_mask = edge_lens < self.CURVE_THRESHES[id] * sqrt_radii[assign_index[0]]
            assign_index = assign_index[:, edge_len_mask]

            # estimate connected components. This is a heuristic for 3D curves
            N, E = this_pnts_trch.size(0), assign_index.size(1)
            adj = to_scipy_sparse_matrix(assign_index, num_nodes=N)
            num_components, component = sp.csgraph.connected_components(adj)
            this_curveidxs = torch.from_numpy(component).long()

            # reorder into contiguous scanning sequence, based on timestep of capture
            curveidx_and_tstep = np.stack([this_curveidxs, this_tsteps.flatten()], axis=1)  # N x 2
            reorder = np.lexsort((curveidx_and_tstep[:, 1], curveidx_and_tstep[:, 0]))
            this_pnts = this_pnts[reorder]
            this_reflectances = this_reflectances[reorder]
            this_labs = this_labs[reorder]
            this_curveidxs = this_curveidxs[reorder]

            # chop at contiguous-issues while traversing along curve
            if this_pnts.shape[0] > 1:
                this_curveidxs = self._partition_curves_at_discontinuities(torch.from_numpy(this_pnts), this_curveidxs, 0.9*self.CURVE_THRESHES[id])
            this_total_curves = this_curveidxs.max().item()
            this_curveidxs += total_curves  # offset by how many curves we have

            # record everything
            per_sensor_pts.append(this_pnts)
            per_sensor_labels.append(this_labs)
            per_sensor_curveidxs.append(this_curveidxs)
            per_sensor_refs.append(this_reflectances)
            total_curves += this_total_curves

        # concatenate per-sensor info
        points = np.concatenate(per_sensor_pts, axis=0)
        labels = np.concatenate(per_sensor_labels, axis=0)
        reflectances = np.concatenate(per_sensor_refs, axis=0)
        curve_idxs = np.concatenate(per_sensor_curveidxs, axis=0)

        # compress curve_idxs to ensure continuously increasing indices
        curve_idxs = np.unique(curve_idxs, return_inverse=True)[1]
        curve_idxs = torch.from_numpy(curve_idxs)
        return curve_idxs, points, labels, reflectances

    @staticmethod
    def _partition_curves_at_discontinuities(points, curve_idxs, thresh):
        # Our curves may not follow continuity requirements as we traverse the curve
        edge_splits = (curve_idxs[1:] - curve_idxs[:-1]) != 0

        def compute_edge_lens(p, edge_spacing=1):
            e = p[edge_spacing:].double() - p[:-edge_spacing].double()
            e_lens = torch.linalg.norm(e, dim=-1)
            e_lens = torch.cat([torch.ones(edge_spacing)*10, e_lens])
            return e_lens

        # compute each edge-length along a curve
        edge_lengths = compute_edge_lens(points, edge_spacing=1)
        edge_lengths_skip = compute_edge_lens(points, edge_spacing=2)

        # identify where curves are discontinuous
        sqrt_radii_thresh = torch.sqrt(torch.linalg.norm(points[:, :2], dim=-1))
        curve_splits_3D = (edge_lengths > (thresh * sqrt_radii_thresh)) & (edge_lengths_skip > (thresh * sqrt_radii_thresh))
        curve_splits = curve_splits_3D | torch.cat([torch.zeros(1, dtype=torch.bool), edge_splits])

        # broadcast into 3D curve indices
        curve_idxs = torch.cumsum(curve_splits, dim=0) - 1
        return curve_idxs


# for debugging
if __name__ == '__main__':
    wandb.init('audi_dataset')
    dataset_config_file = '/home/colton/Documents/curvenet/configs/semantic_A2D2.yaml'
    data_path = '/home/colton/Documents/data/A2D2_dataset/A2D2_dataset'
    val_dataset = SemA2D2(data_path, dataset_config_file, split='val')
    val_dataset.get(100, True)