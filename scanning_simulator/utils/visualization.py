from matplotlib import pyplot as plt
import plotly.graph_objects as go
import cv2
import numpy as np

import torch.utils.data.dataloader
from src.visualization.mitsuba_render import render_pc_shapenet


def visualize_shapenet_pc(pc_ours, segmentation_ours, pc_orig, segmentation_orig, viz_mitsuba=False, counter=0):
    if viz_mitsuba:
        R_x = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]],  dtype=torch.double).cuda()
        pc_viz = pc_ours.cuda().double() @ R_x.T
        pc_viz += torch.tensor([[0.5, 0.5, 0.5]]).cuda()
        viz_clrs_nocs = pc_viz.clone()
        viz_clrs_nocs[:, 0] += 0.125
        viz_clrs_nocs = torch.clip(viz_clrs_nocs * 1.0 + torch.tensor([[0.00, 0.00, 0.00]]).cuda(), 0, 1)
        segmentation_orig = segmentation_ours.cpu().flatten().numpy()
        cmap = plt.get_cmap('Set2')  # cool, viridis, rainbow
        viz_clrs = np.unique(segmentation_orig, return_inverse=True)[1] / np.unique(segmentation_orig).shape[0]
        viz_clrs = cmap(viz_clrs)[:, :3]
        viz_clrs = np.clip(viz_clrs * 1.0 + np.array([[0.00, 0.00, 0.00]]), 0, 1)

        # combine
        pc_viz = pc_viz.cpu().numpy()
        viz_clrs = viz_clrs[:, [1, 0, 2]]
        img = render_pc_shapenet(pc_viz, viz_clrs, point_radius=0.01, width=1000, height=1000)
        img = np.array(img)
        cv2.imwrite("./nocs_seg_%s.png" % counter, img*255)
        img2 = render_pc_shapenet(pc_viz, viz_clrs_nocs, point_radius=0.01, width=1000, height=1000)
        img2 = np.array(img2)
        cv2.imwrite("./nocs_%s.png" % counter, img2*255)

    else:
        pc_ours, segmentation_ours, pc_orig, segmentation_orig = pc_ours.cpu().numpy(), segmentation_ours.cpu().numpy(), pc_orig.cpu().numpy(), segmentation_orig.cpu().numpy()
        # get colors for our pc and the original ShapeNet PC
        cmap = plt.get_cmap('tab20b')  # cool, viridis, rainbow
        seg_clrs = np.unique(segmentation_ours, return_inverse=True)[1] / np.unique(segmentation_ours).shape[0]
        clrs = cmap(seg_clrs)[:, :3] * 255
        cmap = plt.get_cmap('tab20c')  # cool, viridis, rainbow
        orig_seg_clrs = np.unique(segmentation_orig, return_inverse=True)[1] / np.unique(segmentation_orig).shape[0]
        orig_clrs = cmap(orig_seg_clrs)[:, :3] * 255
        clrs = np.concatenate([clrs, orig_clrs])
        pc = np.concatenate([pc_ours, pc_orig], axis=0)

        clrs = [f'rgb({int(clrs[i, 0])}, {int(clrs[i, 1])}, {int(clrs[i, 2])})' for i in range(clrs.shape[0])]
        fig1 = go.Figure(data=[go.Scatter3d(x=pc[:, 0],
                                            y=pc[:, 1],
                                            z=pc[:, 2],
                                            mode='markers',
                                            marker=dict(color=clrs, size=2))])
        fig1.update_layout(title_text="Shape Segmentation",
                           scene=dict(
                               xaxis=dict(nticks=4, range=[-0.5, 0.5], ),
                               yaxis=dict(nticks=4, range=[-0.5, 0.5], ),
                               zaxis=dict(nticks=4, range=[-0.5, 0.5], ), ),
                           scene_aspectmode='cube',
                           )
        fig1.write_html("./fig_seg_%s.html" % counter)


def visualize_kortx_pc(pc_ours, segmentation_ours, pc_orig, segmentation_orig, counter):
    # get colors for our pc and the original ShapeNet PC
    cmap = plt.get_cmap('tab20b')  # cool, viridis, rainbow
    seg_clrs = np.unique(segmentation_ours, return_inverse=True)[1] / np.unique(segmentation_ours).shape[0]
    clrs = cmap(seg_clrs)[:, :3] * 255
    cmap = plt.get_cmap('tab20c')  # cool, viridis, rainbow
    orig_seg_clrs = np.unique(segmentation_orig, return_inverse=True)[1] / np.unique(segmentation_orig).shape[0]
    orig_clrs = cmap(orig_seg_clrs)[:, :3] * 255
    clrs = np.concatenate([clrs, orig_clrs])

    # translate shapenet PC to be next to us
    # pc_orig += np.array([[1, 0, 0]])
    pc = np.concatenate([pc_ours, pc_orig], axis=0)

    clrs = [f'rgb({int(clrs[i, 0])}, {int(clrs[i, 1])}, {int(clrs[i, 2])})' for i in range(clrs.shape[0])]
    fig1 = go.Figure(data=[go.Scatter3d(x=pc[:, 0],
                                        y=pc[:, 1],
                                        z=pc[:, 2],
                                        mode='markers',
                                        marker=dict(color=clrs, size=2))])
    fig1.update_layout(title_text="Shape Segmentation",
                       scene=dict(
                           xaxis=dict(nticks=4, range=[-0.5, 0.5], ),
                           yaxis=dict(nticks=4, range=[-0.5, 0.5], ),
                           zaxis=dict(nticks=4, range=[-0.5, 0.5], ), ),
                       scene_aspectmode='cube',
                       )
    fig1.write_html("./fig_seg_%s.html" % counter)


def visualize_kortx_pc_mitsuba(pnts, curve_idxs, idx, sub_index, point_radius=0.001, use_vivid=False):
    if use_vivid:
        curve_reds = [float(hash(str(idx) + 'rd') % 256) / 255 for idx in curve_idxs.tolist()]
        curve_greens = [float(hash(str(idx) + 'grn') % 256) / 255 for idx in curve_idxs.tolist()]
        curve_blues = [float(hash(str(idx) + 'bl') % 256) / 255 for idx in curve_idxs.tolist()]
        clrs = np.stack([curve_reds, curve_greens, curve_blues], axis=1)
    else:
        cmap = plt.get_cmap('Dark2')  # cool, viridis, rainbow
        seg_clrs = torch.unique(curve_idxs.flatten(), return_inverse=True)[1] / torch.unique(curve_idxs.flatten()).size(0)
        clrs = cmap(seg_clrs.cpu().numpy())[:, :3]
    clrs = torch.from_numpy(clrs)

    R_x = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]],  dtype=torch.double).cuda()
    R_x_2 = torch.tensor([[1, 0, 0], [0, np.cos(np.pi/7), -np.sin(np.pi/7)], [0, np.sin(np.pi/7), np.cos(np.pi/7)]],  dtype=torch.double).cuda()

    pnts = pnts.cuda().double() @ R_x.T @ R_x_2.T + 0.5 + torch.tensor([[0, 0, 0.2]]).cuda()

    if clrs is None:
        clrs = pnts.clone()
        clrs[:, 0] += 0.125
        clrs = torch.clip(clrs * 1.0 + torch.tensor([[0.00, 0.00, 0.00]]).cuda(), 0, 1)

    # combine
    viz_pts = torch.cat([pnts], dim=0).cpu().numpy() + np.array([[0, 0, -0.3]])
    viz_clrs = torch.cat([clrs], dim=0).cpu().numpy()
    viz_clrs = viz_clrs[:, [1, 0, 2]]

    img = render_pc_shapenet(viz_pts, viz_clrs, point_radius=point_radius, width=1000, height=1000, sample_count=64, camera_origin=(10/4, 10.0/4.0, 22.0/4.0))
    img = np.array(img)
    # img = (img*255).astype(np.uint8)
    cv2.imwrite("./object_%s_%s.png" % (idx, sub_index), img*255)


def visualize_kortx_test_pc_mitsuba(pnts, curve_idxs, idx, sub_index, point_radius=0.001):
    curve_reds = [float(hash(str(idx) + 'r') % 256) / 255 for idx in curve_idxs.tolist()]
    curve_greens = [float(hash(str(idx) + 'g') % 256) / 255 for idx in curve_idxs.tolist()]
    curve_blues = [float(hash(str(idx) + 'b') % 256) / 255 for idx in curve_idxs.tolist()]
    clrs = np.stack([curve_reds, curve_greens, curve_blues], axis=1)
    clrs = torch.from_numpy(clrs)

    R_x = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]],  dtype=torch.double).cuda()
    R_x_2 = torch.tensor([[1, 0, 0], [0, np.cos(np.pi/7), -np.sin(np.pi/7)], [0, np.sin(np.pi/7), np.cos(np.pi/7)]],  dtype=torch.double).cuda()

    pnts = pnts.cuda().double() @ R_x.T @ R_x_2.T + 0.5 + torch.tensor([[0, 0, 0.2]]).cuda()

    if clrs is None:
        clrs = pnts.clone()
        clrs[:, 0] += 0.125
        clrs = torch.clip(clrs * 1.0 + torch.tensor([[0.00, 0.00, 0.00]]).cuda(), 0, 1)

    # combine
    viz_pts = torch.cat([pnts], dim=0).cpu().numpy()
    viz_clrs = torch.cat([clrs], dim=0).cpu().numpy()
    viz_clrs = viz_clrs[:, [1, 0, 2]]

    img = render_pc_shapenet(viz_pts, viz_clrs, point_radius=point_radius, width=1200, height=1200, sample_count=64, camera_origin=(3.0, 9.0/4.0, 9.0/4.0))
    img = np.array(img)
    # img = (img*255).astype(np.uint8)
    cv2.imwrite("./nocs_%s_%s_v2.png" % (idx, sub_index), img*255)


def visualize_kortx_test_pc(pc_ours, segmentation_ours, counter):
    # get colors for our pc and the original ShapeNet PC
    cmap = plt.get_cmap('Set1')  # cool, viridis, rainbow
    seg_clrs = np.unique(segmentation_ours, return_inverse=True)[1] / np.unique(segmentation_ours).shape[0]
    seg_clrs = (seg_clrs * 1317.1) % 1
    clrs = cmap(seg_clrs)[:, :3] * 255

    # translate shapenet PC to be next to us
    pc = np.concatenate([pc_ours], axis=0)

    clrs = [f'rgb({int(clrs[i, 0])}, {int(clrs[i, 1])}, {int(clrs[i, 2])})' for i in range(clrs.shape[0])]
    fig1 = go.Figure(data=[go.Scatter3d(x=pc[:, 0],
                                        y=pc[:, 1],
                                        z=pc[:, 2],
                                        mode='markers',
                                        marker=dict(color=clrs, size=2))])
    fig1.update_layout(title_text="Shape Segmentation",
                       scene=dict(
                           xaxis=dict(nticks=4, range=[-0.5, 0.5], ),
                           yaxis=dict(nticks=4, range=[-0.5, 0.5], ),
                           zaxis=dict(nticks=4, range=[-0.5, 0.5], ), ),
                       scene_aspectmode='cube',
                       )
    fig1.write_html("./fig_seg_%s.html" % counter)
    # fig1.show()
    # wandb.log({"Segmentation-Sample": fig1})
