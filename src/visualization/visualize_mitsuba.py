from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import wandb
import torch
from src.visualization.mitsuba_render import render_pc_shapenet, render_nocs
import cv2


def visualize_nocs_mitsuba(pc, pred_nocs, gt_nocs, errs, max_err=0.1, title="NOCS L1 Error Vis"):
    pc, pred_nocs, gt_nocs, errs = pc.to('cpu').data.numpy(), pred_nocs.to('cpu').data.numpy(), gt_nocs.to('cpu').data.numpy(), errs.to('cpu').data.numpy()

    # build PC
    pred_nocs += np.array([[0.5, 0.5, 0.5]])
    gt_nocs += np.array([[0.5, 0.5, 0.5]])

    # get colors
    err_cmap = plt.get_cmap('plasma')
    errs /= max_err
    clrs_pred = err_cmap(errs)[:, :3]
    clrs_pc = np.ones((pc.shape[0], 3)) * 0.55
    clrs_nocs = np.clip(gt_nocs[:, :3], 0, 1)

    # create pyplot for 3 images
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    render_pc_shapenet(pc, clrs_pc, ax=axes[0])
    render_nocs(pred_nocs, clrs_pred, ax=axes[1])
    render_nocs(gt_nocs, clrs_nocs, ax=axes[2])
    plt.axis('off')
    plt.tight_layout()
    wandb.log({title: fig})
    plt.close(fig)
    # fig.show()


def visualize_seg_mitsuba(pc, pred_labels, gt_labels, errs, correct_mask, title="Segmentation Vis", max_err=1.0, camera_origin=(2.2, 2.2, 2.2), is_kortx=False, index=0):
    pc, pred_labels, gt_labels, errs , correct_mask= pc.to('cpu').data.numpy(), pred_labels.to('cpu').data.numpy(), gt_labels.to('cpu').data.numpy(), errs.to('cpu').data.numpy(), correct_mask.to('cpu').data.numpy()
    pc = (pc * 0.8) + np.array([[0.5, 0.5, 0.5]])

    # prediction colors
    seg_cmap = plt.get_cmap("Dark2")
    max_parts = 5
    pred_seg_clrs = seg_cmap(pred_labels / max_parts)[:, :3]
    gt_seg_clrs = seg_cmap(gt_labels / max_parts)[:, :3]

    # correct-prediction colors
    clrs_correct = np.ones((pc.shape[0], 3)) * 0.55
    clrs_correct[~correct_mask] = np.array([1.0, 0, 0])

    # create pyplot for 4 images
    im1 = render_pc_shapenet(pc, pred_seg_clrs, ax=None, width=800, height=800, camera_origin=camera_origin, is_kortx=is_kortx, sample_count=64, point_radius=0.004)
    im1 = np.array(im1 ** (1.0 / 2.2))
    im2 = render_pc_shapenet(pc, gt_seg_clrs, ax=None, width=800, height=800, camera_origin=camera_origin, is_kortx=is_kortx, sample_count=64, point_radius=0.004)
    im2 = np.array(im2 ** (1.0 / 2.2))
    full_img = [im1, im2]
    full_img = np.concatenate(full_img, axis=1)

    cv2.imwrite(title+"%03d.png" % index, np.clip(full_img * 255, 0, 255).astype(int)[:, :, [2, 1, 0]])


def visualize_pc_mitsuba(pc, clr_vals, title="Visualized Point Cloud", shift=[0.0, 0.0, 0.0]):
    pc, clr_vals = pc.to('cpu').data.numpy(), clr_vals.to('cpu').data.numpy()
    pc += np.array([shift])

    # get colors
    clr_cmap = plt.get_cmap('coolwarm')
    clrs = clr_vals - np.min(clr_vals)
    clrs /= (np.mean(clrs) + np.std(clrs)*2.5)
    clrs = clr_cmap(clrs)[:, :3]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(25, 25))

    # Add in color bar
    # fig.subplots_adjust(right=0.5)
    cmap = plt.get_cmap('coolwarm')
    norm = mpl.colors.Normalize(vmin=np.min(clr_vals), vmax=np.max(clr_vals))
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=None, orientation='vertical', label='PCA Value')

    # Add in point cloud render
    render_pc_shapenet(pc, clrs, ax=ax)
    plt.axis('off')
    plt.tight_layout()
    wandb.log({title: fig})
    # fig.show()


