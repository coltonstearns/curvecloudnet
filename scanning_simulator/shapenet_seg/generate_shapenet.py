import numpy as np
import torch
import argparse
import tqdm
import os

from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
import torch.utils.data.dataloader

from scanning_simulator.utils.sampling import SampleMeshes
from scanning_simulator.shapenet_seg.shapenet_seg_dataset import ShapeNetSegDataset
from scanning_simulator.shapenet_seg.shapenet_dataset import ShapeNetCoreWithSplit
from scanning_simulator.utils.visualization import visualize_shapenet_pc


CATEGORY_IDS = {
    'Airplane': '02691156',
    'Bag': '02773838',
    'Cap': '02954340',
    'Car': '02958343',
    'Chair': '03001627',
    'Earphone': '03261776',
    'Guitar': '03467517',
    'Knife': '03624134',
    'Lamp': '03636649',
    'Laptop': '03642806',
    'Motorbike': '03790512',
    'Mug': '03797390',
    'Pistol': '03948459',
    'Rocket': '04099429',
    'Skateboard': '04225987',
    'Table': '04379243',
}

CATEGORY_NAMES = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop',
                  'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
LABEL_IDS = {CATEGORY_NAMES[i]: i for i in range(len(CATEGORY_NAMES))}


# CAMERA_VIEWPOINT = (1.0, -np.pi + 0.2, 0.0, 0.0)  # front-on for ONet-Watertight
# CAMERA_VIEWPOINT = (1.3, np.pi+np.pi/6, -np.pi/2, 0.0)  # front-on for ShapeNetCore1.0
CAMERA_VIEWPOINT = (1.3, np.pi+np.pi/6, -np.pi/2 + np.pi/4, 0.0)  # corner nocs for ShapeNetCore1.0
# np.pi puts in correct orientation


def main(args, device):
    mesh_vis_freq = 10
    shapenet_seg_dataset = ShapeNetSegDataset(args.seg_data_path, args.category, split=args.split)
    if args.category != "all":
        synset_id = shapenet_seg_dataset.category_ids[args.category]
    else:
        synset_id = None
    shapenet_mesh_dataset = ShapeNetCoreWithSplit(args.mesh_data_path, version=1, texture_resolution=1,
                                             synsets=synset_id, load_textures=False, split=args.split)

    # go through dataset and sample points
    dataset_points, dataset_normals, dataset_3d_idxs, dataset_curvatures = [], [], [], []
    dataset_labels, dataset_segmentations = [], []
    mesh_sampler = SampleMeshes(args.npoints, device, resolution=args.raster_res, camera_view=CAMERA_VIEWPOINT, scan_style=args.scan_style,
                                scan_direction=args.scan_direction, line_density=args.scanline_density)

    for i, data in enumerate(tqdm.tqdm(shapenet_seg_dataset)):
        # sample points from mesh
        model_id = data.model_id
        synset_id = getattr(data, 'synset_id') if hasattr(data, 'synset_id') else CATEGORY_IDS[args.category]

        # load model data
        try:
            model = [shapenet_mesh_dataset.load_obj_file_by_model_idx(synset_id, model_id)]
        except (FileNotFoundError, ValueError) as e:
            print("Model not found: %s, %s" % (synset_id, model_id))
            print(e)
            continue

        # extract labels, vertices, and faces
        labels = [LABEL_IDS[model[j]['label']] for j in range(len(model))]
        vertices = [model[j]['verts'].to(device).float() for j in range(len(model))]
        faces = [model[j]['faces'].to(device).float() for j in range(len(model))]

        # perform point sampling (save some of the mesh renders for visualization)
        viz_outdir = args.mesh_viz_outdir if (i % mesh_vis_freq) == 0 else None
        curve_clouds = mesh_sampler.meshes2points(vertices, faces, labels, viz_outdir=viz_outdir)
        if curve_clouds is None:  # failure somewhere in preprocessing
            print("!!! FAILED TO CREATE CURVE CLOUD !!!")
            continue

        # get sampled point segmentations
        closest_idxs = torch.argmin(torch.cdist(curve_clouds.points[0].double(), data.pos.to(device).double()), dim=1)
        segmentation = data.y.to(device)[closest_idxs]
        if args.viz:
            visualize_shapenet_pc(curve_clouds.points[0], segmentation, data.pos, data.y, args.viz_mitsuba, counter=i)

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
    parser = argparse.ArgumentParser(description='Preprocess ShapeNetCoreV1 into segmentation and classification task.')
    parser.add_argument('--seg-data-path', type=str, help='Base directory to shapenet segmentation dataset')
    parser.add_argument('--mesh-data-path', type=str, help='Base directory to shapenet core V1 dataset')
    parser.add_argument('--outdir', type=str, default='./', help='Filename to output preprocessed data to.')
    parser.add_argument('--category', type=str, default='all', help='Class names to use. Look at ShapeNet for a complete list.')
    parser.add_argument('--npoints', type=int, default=2048, help='Approximate number of points to sample.')
    parser.add_argument('--split', type=str, default="train", help='Official ShapeNet split. One of ["train", "val", "test"]')
    parser.add_argument('--raster-res', type=int, default=2048, help='Resolution of rasterization. Points are sampled from this image plane.')
    parser.add_argument('--scanline-density', type=float, default=0.25, help='Max density is 1.0. Downsamples lines by this factor below it (from original resolution).')
    parser.add_argument('--scan-style', type=str, default='linear', help='Point sampling pattern. One of ["linear", "sine"]')
    parser.add_argument('--scan-direction', type=str, default='random', help='Point sampling direction. One of ["parallel", "grid", "random"]')

    parser.add_argument('--viz', action='store_true', help='Visualize preprocessed point clouds.')
    parser.add_argument('--mesh_viz_outdir', type=str, help='Directory to output renderings of mesh faces.')
    parser.add_argument('--viz_mitsuba', action='store_true', help='Visualize preprocessed point clouds.')

    args = parser.parse_args()
    device = "cuda:0"

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    if args.mesh_viz_outdir is not None:
        if not os.path.exists(args.mesh_viz_outdir):
            os.makedirs(args.mesh_viz_outdir)
    args.outfile = os.path.join(args.outdir, "data_%s_%s_%s_%s_%s.pth" % (args.npoints, args.raster_res, args.scanline_density, args.scan_direction, args.split))
    main(args, device)

