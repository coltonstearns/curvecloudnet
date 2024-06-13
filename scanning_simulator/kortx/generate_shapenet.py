import numpy as np
import torch
import argparse
import tqdm
import os

from scanning_simulator.utils.sampling import SampleMeshes
from scipy.spatial.transform import Rotation as R
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
import torch.utils.data.dataloader

from scanning_simulator.shapenet_seg.shapenet_seg_dataset import ShapeNetSegDataset
from scanning_simulator.shapenet_seg.shapenet_dataset import ShapeNetCoreWithSplit
from scanning_simulator.utils.visualization import visualize_kortx_pc, visualize_kortx_pc_mitsuba


CATEGORY_NAMES = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop',
                  'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
LABEL_IDS = {CATEGORY_NAMES[i]: i for i in range(len(CATEGORY_NAMES))}

# CAMERA_VIEWPOINT = (1.0, -np.pi + 0.2, 0.0, 0.0)  # front-on for ONet-Watertight
CAMERA_VIEWPOINT = (1.3, np.pi+np.pi/10, -np.pi/2, 0.0)  # front-on for ShapeNetCore1.0
# CAMERA_VIEWPOINT = (1.3, np.pi+np.pi/6, -np.pi/2 + np.pi/4, 0.0)  # corner nocs for ShapeNetCore1.0


def rotate_augment(pc):
    rot_mat = torch.tensor(R.random().as_matrix()).to(pc)
    pc_aug = pc @ rot_mat.T
    return pc_aug, rot_mat


def main(args, device):
    mesh_vis_freq = 5

    # load ShapeNet Datasets
    args.category = "all" if "all" in args.category else args.category
    shapenet_seg_dataset = ShapeNetSegDataset(args.seg_data_path, args.category, split=args.split)

    if args.category == "all":
        synset_ids = None
    else:
        synset_ids = [shapenet_seg_dataset.category_ids[args.category[i]] for i in range(len(args.category))]
    shapenet_mesh_dataset = ShapeNetCoreWithSplit(args.mesh_data_path, version=1, texture_resolution=1,
                                             synsets=synset_ids, load_textures=False, split=args.split)

    # go through dataset and sample points
    dataset_points, dataset_normals, dataset_3d_idxs, dataset_curvatures = [], [], [], []
    dataset_labels, dataset_segmentations = [], []
    mesh_sampler = SampleMeshes(args.npoints, device, resolution=args.raster_res, camera_view=CAMERA_VIEWPOINT, scan_style=args.scan_style,
                                scan_direction=args.scan_direction, line_density=args.scanline_density)

    for repeat in range(args.num_dataset_repeats):
        print("On iteration %s of %s..." % (repeat + 1, args.num_dataset_repeats))
        for i, data in enumerate(tqdm.tqdm(shapenet_seg_dataset)):
            # sample points from mesh
            model_id = data.model_id
            synset_id = data.synset_id
            try:
                model = [shapenet_mesh_dataset.load_obj_file_by_model_idx(synset_id, model_id)]
            except (FileNotFoundError, ValueError) as e:
                continue

            labels = [LABEL_IDS[model[j]['label']] for j in range(len(model))]
            vertices = [model[j]['verts'].to(device).float() for j in range(len(model))]
            faces = [model[j]['faces'].to(device).float() for j in range(len(model))]

            # augment mesh by random SE3 transformation
            vertices_aug, rots = [], []
            for j in range(len(vertices)):
                se3_aug = rotate_augment(vertices[j])
                vertices_aug.append(se3_aug[0])
                rots.append(se3_aug[1])

            # perform point sampling
            viz_outdir = args.mesh_viz_outdir if (i % mesh_vis_freq) == 0 else None
            curve_clouds = mesh_sampler.meshes2points(vertices_aug, faces, labels, viz_outdir=viz_outdir)
            if curve_clouds is None:  # failure somewhere in preprocessing
                continue

            # get sampled point segmentations
            canonical_pnts = curve_clouds.points[0].double() @ rots[0].double()
            closest_idxs = torch.argmin(torch.cdist(canonical_pnts, data.pos.to(device).double()), dim=1)
            segmentation = data.y.to(device)[closest_idxs]

            # put into same coordinate frame as summer robotics data
            curve_clouds.points[:, :, [0, 2]] *= -1
            curve_clouds.points[:, :, [0, 2]] = curve_clouds.points[:, :, [2, 0]]
            curve_clouds.points[:, :, 0] *= -1
            curve_clouds.normals[:, :, [0, 2]] *= -1
            curve_clouds.normals[:, :, [0, 2]] = curve_clouds.points[:, :, [2, 0]]
            curve_clouds.normals[:, :, 0] *= -1

            # rotate 45 degrees to make more similar to summer robotics data
            angle = 45 / 180 * np.pi
            rot_mat_45 = torch.tensor([[np.cos(angle), -np.sin(angle), 0],
                                       [np.sin(angle), np.cos(angle), 0],
                                       [0, 0, 1]]).to(curve_clouds.points)
            curve_clouds.points = curve_clouds.points @ rot_mat_45.T.view(1, 3, 3)

            if args.viz:
                segmentation = torch.unique(segmentation, return_inverse=True)[1]
                # visualize_kortx_pc(curve_clouds.points[0].cpu().numpy(), segmentation.cpu().numpy(), data.pos.cpu().numpy(), data.y.cpu().numpy(), counter=i)
                visualize_kortx_pc_mitsuba(curve_clouds.points[0].cpu(), curve_clouds.point2curve_idxs[0].cpu(), idx=i, sub_index=repeat, point_radius=0.003)
                visualize_kortx_pc_mitsuba(curve_clouds.points[0].cpu(), segmentation.cpu(), idx=i, sub_index=repeat+1, point_radius=0.003)

            # record sampled point data
            dataset_points += [curve_clouds.points[i] for i in range(len(curve_clouds))]
            dataset_normals += [curve_clouds.normals[i] for i in range(len(curve_clouds))]
            dataset_3d_idxs += [curve_clouds.point2curve_idxs[i] for i in range(len(curve_clouds))]
            dataset_curvatures += [curve_clouds.curvature[i] for i in range(len(curve_clouds))]
            dataset_segmentations += [segmentation]
            dataset_labels += labels

    # format as torch_geometric object
    dataset_labels = torch.tensor(dataset_labels).long()
    torch_geo_datalist = [Data(x=None, pos=dataset_points[j].cpu(), y=dataset_segmentations[j].cpu(), labels=dataset_labels[j].cpu(),
                               normals=dataset_normals[j].cpu(),
                               curve_idxs=dataset_3d_idxs[j], curvature=dataset_curvatures[j]) for j in range(len(dataset_points))]
    formatted_data = Batch.from_data_list(torch_geo_datalist)
    with open(args.outfile, "wb") as f:
        torch.save(formatted_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess ShapeNetCoreV1 into Kortx training set.')
    parser.add_argument('--seg-data-path', type=str, help='Base directory to shapenet segmentation dataset')
    parser.add_argument('--mesh-data-path', type=str, help='Base directory to shapenet core V1 dataset')
    parser.add_argument('--outdir', type=str, default='./', help='Filename to output preprocessed data to.')
    parser.add_argument('--category', type=str, nargs='+', default='all', help='Class names to use. Look at ShapeNet for a complete list.')
    parser.add_argument('--npoints', type=int, default=2048, help='Approximate number of points to sample.')
    parser.add_argument('--split', type=str, default="train", help='Official ShapeNet split. One of ["train", "val", "test"]')
    parser.add_argument('--raster-res', type=int, default=2048, help='Resolution of rasterization. Points are sampled from this image plane.')
    parser.add_argument('--scanline-density', type=float, default=0.1666, help='Max density is 1.0. Downsamples lines by this factor below it (from original resolution).')
    parser.add_argument('--scan-style', type=str, default='linear', help='Point sampling pattern. One of ["linear", "sine"]')
    parser.add_argument('--scan-direction', type=str, default='random', help='Point sampling direction. One of ["parallel", "grid", "random"]')
    parser.add_argument('--num-dataset-repeats', type=int, default=5, help='Number of time to go through ShapeNet (because we randomly rotate each time)')

    parser.add_argument('--viz', action='store_true', help='Visualize preprocessed point clouds.')
    parser.add_argument('--mesh_viz_outdir', type=str, help='Directory to output renderings of mesh faces.')

    args = parser.parse_args()
    device = "cuda:0"

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    if args.mesh_viz_outdir is not None:
        if not os.path.exists(args.mesh_viz_outdir):
            os.makedirs(args.mesh_viz_outdir)
    args.outfile = os.path.join(args.outdir, "data_%s_%s_%s_%s_%s.pth" % (args.npoints, args.raster_res, args.scanline_density, args.scan_direction, args.split))
    main(args, device)
