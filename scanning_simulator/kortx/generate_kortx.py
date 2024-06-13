import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
import torch.utils.data.dataloader
import argparse
import os
import json
import pandas as pd
from scanning_simulator.utils.visualization import visualize_kortx_test_pc, visualize_kortx_test_pc_mitsuba


CATEGORY_NAMES = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop',
                  'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
LABEL_IDS = {CATEGORY_NAMES[i]: i for i in range(len(CATEGORY_NAMES))}


# CAMERA_VIEWPOINT = (1.0, -np.pi + 0.2, 0.0, 0.0)  # front-on for ONet-Watertight
CAMERA_VIEWPOINT = (1.3, np.pi+np.pi/10, -np.pi/2, 0.0)  # front-on for ShapeNetCore1.0
# CAMERA_VIEWPOINT = (1.3, np.pi+np.pi/6, -np.pi/2 + np.pi/4, 0.0)  # corner nocs for ShapeNetCore1.0


def main(args, device):
    # process each scanning-view of the object
    dataset_points, dataset_normals, dataset_curve_idxs, dataset_curvatures = [], [], [], []
    dataset_labels, dataset_segmentations = [], []
    for ii, instance_id in enumerate(os.listdir(args.data_path)):
        with open(os.path.join(args.data_path, instance_id, "info.json"), 'r') as f:
            json_data = json.load(f)
            object_label = LABEL_IDS[json_data['object-class']]
            capture_setting = json_data['capture-setting']

        for k, view_fname in enumerate(os.listdir(os.path.join(args.data_path, instance_id))):
            # skip if its not one of our view data files
            if not view_fname.endswith('.csv') or 'background' in view_fname:
                continue

            df = pd.read_csv(os.path.join(args.data_path, instance_id, view_fname), header=None)
            total_pnts = df.shape[0]

            # get sample starts idxs
            interval = (total_pnts - args.npoints) // args.samples_per_scan
            start_idxs = torch.arange(args.samples_per_scan).view(-1, 1) * interval
            idxs = (start_idxs + torch.arange(args.npoints).view(1, -1)).flatten()

            # get full shapenet object info
            all_pnts = torch.from_numpy(df.iloc[:, :3].to_numpy())
            all_times = torch.from_numpy(df.iloc[:, 3].to_numpy()).long()
            all_curve_idxs = torch.from_numpy(df.iloc[:, 4].to_numpy()).long()
            all_segmentation = torch.from_numpy(df.iloc[:, 5].to_numpy()).long()

            # fix curve idxs (incorrect processing in previous step)
            all_curve_idxs = fix_curve_idxs(all_curve_idxs)

            # cut curves at large thresholds
            all_curve_idxs = cut_curves(all_curve_idxs, all_pnts, thresh=0.2 if capture_setting == 'table' else 1.0)

            # subsample to curves
            pnts = all_pnts[idxs].view(args.samples_per_scan, args.npoints, 3)
            times = all_times[idxs].view(args.samples_per_scan, args.npoints)
            time_spans = times.max(dim=1)[0] - times.min(dim=1)[0]
            max_vals = pnts.abs().view(args.samples_per_scan, -1).max(dim=1)[0].view(args.samples_per_scan, 1, 1) * 2
            pnts /= max_vals
            curve_idxs = all_curve_idxs[idxs].view(args.samples_per_scan, args.npoints)
            segmentation = all_segmentation[idxs].view(args.samples_per_scan, args.npoints)

            # squeeze curve idxs into continuous subspace
            curve_idxs = [torch.unique(curve_idxs[i], return_inverse=True)[1] for i in range(curve_idxs.size(0))]
            if args.viz:
                visualize_kortx_test_pc(pnts[0].cpu().numpy(), curve_idxs[0].cpu().numpy(), counter=len(os.listdir(args.data_path)) * ii + k)
                visualize_kortx_test_pc_mitsuba(pnts[0].cpu(), curve_idxs[0].cpu().numpy(), idx=instance_id, sub_index=k, point_radius=0.003)

            # record sampled point data
            dataset_points += [pnts[i] for i in range(pnts.size(0))]
            dataset_curve_idxs += curve_idxs
            dataset_segmentations += [segmentation[i] for i in range(pnts.size(0))]
            dataset_curvatures += [torch.zeros(pnts[i].size()) for i in range(pnts.size(0))]
            dataset_normals += [torch.zeros(pnts[i].size()) for i in range(pnts.size(0))]
            dataset_labels += [object_label for i in range(pnts.size(0))]

    # format as torch_geometric object
    dataset_labels = torch.tensor(dataset_labels).long()
    torch_geo_datalist = [Data(x=None, pos=dataset_points[j].cpu(), y=dataset_segmentations[j].cpu(), labels=dataset_labels[j].cpu(),
                               normals=dataset_normals[j].cpu(),
                               curve_idxs=dataset_curve_idxs[j], curvature=dataset_curvatures[j]) for j in range(len(dataset_points))]
    formatted_data = Batch.from_data_list(torch_geo_datalist)

    with open(args.outfile, "wb") as f:
        torch.save(formatted_data, f)


def fix_curve_idxs(curve_idxs):
    # convert into binary endpoint form
    curve_endpoints = torch.cat([torch.ones(1).to(curve_idxs), curve_idxs[1:] - curve_idxs[:-1]]) > 0

    # correct adjacent (repeated) endpoints
    adjacent_endpoints = torch.where(curve_endpoints[:-1] & curve_endpoints[1:])[0]
    curve_startidxs = curve_endpoints.clone()
    curve_startidxs[adjacent_endpoints] = False

    # convert back
    curve_endpoints = torch.cumsum(curve_startidxs, dim=0)
    return curve_endpoints


def cut_curves(curve_idxs, pos, thresh=0.01):
    # compute edge lengths
    edge_lens = torch.linalg.norm(pos[1:] - pos[:-1], dim=-1)
    cuts = edge_lens > thresh

    # add in edge cuts
    curve_idxs_binary = (curve_idxs[1:] - curve_idxs[:-1]) > 0
    curve_idxs_binary |= cuts

    # convert back
    dev = pos.device
    curve_idxs = torch.cat([torch.zeros(1).to(dev).long(), torch.cumsum(curve_idxs_binary, dim=0)])
    return curve_idxs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess Kortx datsaet into segmentation test split.')
    parser.add_argument('--data-path', type=str, help='Base directory to raw kortx data')
    parser.add_argument('--outdir', type=str, default='./', help='Filename to output preprocessed data to.')
    parser.add_argument('--npoints', type=int, default=2048, help='Approximate number of points to sample.')
    parser.add_argument('--samples_per_scan', type=int, default=5, help='Each summer robotics scan is long. This is how many samples we generate for each.')
    parser.add_argument('--viz', action='store_true', help='Visualize preprocessed point clouds.')

    args = parser.parse_args()
    device = "cuda:0"

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    args.outfile = os.path.join(args.outdir, "data_%s_%s.pth" % (args.npoints, "summer_robotics_test"))
    main(args, device)
