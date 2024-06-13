import numpy as np
import torch
import cv2
import torch.utils.data.dataloader
from scanning_simulator.utils.minimal_rasterizer import rasterize
from scanning_simulator.utils.scanlines import ScanLineGenerator
from scanning_simulator.utils.curve_cloud import CurveClouds

CATEGORY_NAMES = ['airplane', 'bench', 'bookshelf', 'cabinet', 'car', 'chair', 'sofa', 'table']
LABEL_IDS = {CATEGORY_NAMES[i]: i for i in range(len(CATEGORY_NAMES))}
GLOBAL_INDEX = 0

class SampleMeshes:
    # for shapenet, camera perspective is (1.0, 45, 180). for modelnet, it is (1.0, 45, 0)
    def __init__(self, npoints, device, resolution=1024, camera_view=(1.0, 0, 0, 0), scan_style='linear',
                 scan_direction='random', line_density=1.0, curvature_knn=10):
        """
        Last is up-axis rotation!
        Middle is out-axis rotation
        Starting is
        This class takes in batches of meshes and returns sampled point clouds.
        """
        self.npoints = npoints
        self.device = device
        self.resolution = resolution
        self.camera_view = camera_view
        self.scan_style = scan_style
        self.scan_direction = scan_direction
        self.line_density = line_density
        self.curvature_knn = curvature_knn

        # create rotation and translation
        R_x = torch.tensor(cv2.Rodrigues(np.array([camera_view[1], 0, 0], dtype=float))[0], dtype=torch.float32, device=device)
        R_y = torch.tensor(cv2.Rodrigues(np.array([0, camera_view[2], 0], dtype=float))[0], dtype=torch.float32, device=device)
        R_z = torch.tensor(cv2.Rodrigues(np.array([0, 0, camera_view[3]], dtype=float))[0], dtype=torch.float32, device=device)
        self.R = R_x @ R_y @ R_z
        self.t = torch.tensor([[0, 0, camera_view[0]]], device=device)
        self.t = self.t.to(device)

        # set up scan liner
        self.scan_lines_gen = ScanLineGenerator(resolution, device, scan_style, scan_direction, line_density)

    def meshes2points(self, vertices, faces, labels=None, viz_outdir=None):
        """
        Samples points on meshes
        :param meshes (pytorch3d.structures.Meshes): Batch of meshes
        :return:
        """
        coords, normals, masks = self._render_meshes(vertices, faces, viz_outdir)
        curve_clouds = self._sample_curves(coords, normals, masks)
        return curve_clouds

    def _render_meshes(self, vertices, faces, viz_outdir):
        # loop over meshes and run fast-rasterization
        coords, normals, masks = [], [], []
        for i in range(len(vertices)):
            this_coords, this_normals, this_mask = rasterize(vertices[i], faces[i], self.R, self.t, res=self.resolution, viz_outdir=viz_outdir)
            coords.append(this_coords)
            normals.append(this_normals)
            masks.append(this_mask)

        # stack into batched tensor
        coords = torch.stack(coords)
        normals = torch.stack(normals)
        masks = torch.stack(masks)

        return coords, normals, masks

    def _sample_curves(self, coords, normals, masks):
        # get point uv indices (and scan-line idxs as 3rd dim)
        B = coords.size(0)
        success, scans_uv = self.scan_lines_gen.generate_scan_lines(B, self.npoints, masks)
        if not success:
            return None

        # get coordinate data of scan lines
        batch_indices = torch.arange(B).repeat_interleave(self.npoints).to(self.device)
        polyline_points = coords[batch_indices, scans_uv[:, :, 0].flatten(), scans_uv[:, :, 1].flatten()]
        polyline_normals = normals[batch_indices, scans_uv[:, :, 0].flatten(), scans_uv[:, :, 1].flatten()]
        polyline_points = polyline_points.view(B, self.npoints, 3)
        polyline_normals = polyline_normals.view(B, self.npoints, 3)

        # create a curve cloud object
        curve_clouds = CurveClouds(polyline_points, polyline_normals, scans_uv[:, :, :2], scans_uv[:, :, 2],
                                   anti_alias=True, with_intersections=True, curvature_knn=self.curvature_knn)
        return curve_clouds