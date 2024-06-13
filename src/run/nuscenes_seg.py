import os.path
import torch
import time
import wandb
import tqdm
import numpy as np
import os.path as osp
import cv2
import gc

import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from src.visualization.mitsuba_render import render_pc_kitti
from src.models.utils.lovasz_losses import lovasz_softmax_flat
from src.utils.utils import fast_hist, per_class_iu
from src.run.globals import (NUSCENES_CMAP, NUSCENES_CLASSES, NUSCENES_CLASS_WEIGHTS,
                             NUSCENES_VISUALIZATION_IDXS, NUSCENES_IGNORE_LABEL, LOGGED_ERRORS)


def train(model, dataloader, optimizer, device, start_idx, use_lovasz=False, weighted_ce=False):
    model.train()
    print_loss = 0
    t_init = time.time()

    # compute CE weights
    ce_weights = NUSCENES_CLASS_WEIGHTS if weighted_ce else None
    point_errors = torch.zeros(0).float().to('cpu')
    end_idx = len(dataloader) - start_idx
    optimizer.zero_grad()
    for i, data in enumerate(dataloader):
        data = data.to(device)
        try:
            out = model(data, labels=data.y)
            loss, errs = seg_loss(out, data.y, ignore=NUSCENES_IGNORE_LABEL, use_lovasz=use_lovasz, ce_weights=ce_weights)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        except (RuntimeError, MemoryError) as e:
            print("Error: ")
            print(e)
            torch.cuda.empty_cache()
            gc.collect()
            return False, i + start_idx

        # get errors and record
        these_point_errs = global_mean_pool(errs.view(-1, 1), batch=data.batch).cpu()
        point_errors = torch.cat([point_errors, these_point_errs])
        print_loss += loss.item()

        if (i + 1) % 10 == 0:
            avg_point_err = torch.mean(point_errors[max(0,(i-10)*dataloader.batch_size):i*dataloader.batch_size]).item()
            print(f'[{i+1+start_idx}/{len(dataloader)}] Loss: {print_loss / 10:.4f} '
                  f'Train NLL Error: {avg_point_err:.4f}')
            print("Total Time: %s" % (time.time() - t_init))
            wandb.log({"Train-Loss": print_loss, "Train-Error": avg_point_err})
            print_loss = 0
            t_init = time.time()
        if (i + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        if i > end_idx:
            break

    return True, 0


@torch.no_grad()
def val(model, dataloader, device, use_lovasz=False, test_mode=False, outdir=None):
    print("Evaluating...")
    model.eval()
    total_loss = 0

    unique_label_str = dataloader.dataset.nuscyaml['labels_16']
    unique_label_str = [unique_label_str[k] for k in sorted(unique_label_str.keys())][1:]
    unique_label = np.arange(len(unique_label_str))

    flops, times, mem_use = [], [], []
    ious, categories, histogram_counts = [], [], []
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()

    for i, data in enumerate(tqdm.tqdm(dataloader)):
        # run forward pass through model
        data = data.to(device)
        orig_pos = data.pos.clone()
        outs, all_other_losses = [], []
        for axis_flips in range(4):
            for scales in range(3):
                this_pos = orig_pos.clone()
                if axis_flips == 1:
                    this_pos[:, 0] *= -1
                elif axis_flips == 2:
                    this_pos[:, 1] *= -1
                elif axis_flips == 3:
                    this_pos[:, :2] *= -1

                if scales == 1:
                    this_pos[:, :2] *= 0.95
                elif scales == 2:
                    this_pos[:, :2] *= 1.05

                data.pos = this_pos
                t0 = time.time()
                out = model(data, labels=data.y)
                t1 = time.time()
                times.append(t1 - t0)
                mem_use.append(torch.cuda.max_memory_allocated() / 10 ** 9)
                outs.append(out)

        # average over test-time augmentation
        out = sum(outs) / 12  # average over all predictions
        out = F.log_softmax(out, dim=-1)

        # compute validation loss; use_additional_losses allows for an earlier voxel-wise loss!
        loss, _ = seg_loss(out, data.y, use_lovasz=use_lovasz)
        if len(loss.size()) > 0:
            total_loss = total_loss + loss.item() if loss.size(0) > 0 else total_loss  # loss is sizeless in test split

        # go through predicted labels
        predict_labels = torch.argmax(out, dim=-1)
        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for k, (pred, y) in enumerate(zip(predict_labels.split(sizes), data.y.split(sizes))):
            counts_hist = fast_hist_crop(pred.cpu().data.numpy(), y.cpu().data.numpy(), unique_label)
            histogram_counts.append(counts_hist)

        # record if in testing split
        if test_mode:
            assert outdir is not None
            pred_labels_test = torch.argmax(out[:, 1:], dim=-1) + 1  # can't predict "no-class"
            folder_name = osp.join(outdir, "results", "lidarseg", "test")
            if not osp.exists(folder_name):
                os.makedirs(folder_name)
            for k, (pred, reorder) in enumerate(zip(pred_labels_test.split(sizes), data.reorder.split(sizes))):
                pred = pred[reorder].cpu().data.numpy().astype(np.uint8)
                pathname = osp.join(folder_name, data.fname[k])
                pred.tofile(pathname)

    # compute per-class IOU
    per_class_iou = per_class_iu(sum(histogram_counts))
    miou = np.nanmean(per_class_iou)

    # record statistics
    mean_flops = np.mean(flops)
    mean_runtime = np.mean(times[5:])
    memuse = np.mean(mem_use)

    print("========== Validation Results ===========")
    print("Class IOU:")
    print(unique_label_str)
    print(per_class_iou)
    print("mIOU: %s" % miou)
    print("=========================================")

    # Log wandb values
    LOGGED_ERRORS.append(miou)
    median_err_topk = np.median(sorted(LOGGED_ERRORS)[-5:])
    class_ious = {unique_label_str[i]: per_class_iou[i] for i in range(len(unique_label_str))}
    wandb.log({"mIOU": miou, "FLOPS": mean_flops, "Loss": total_loss,
               "Median Top 5 IOU": median_err_topk, "Runtime": mean_runtime, "GPU Memory Allocated": memuse})
    wandb.log(class_ious)
    return miou


@torch.no_grad()
def viz_seg(model, dataloader, device, num_samples, use_lovasz=False, is_kortx=False):
    print("Visualizing...")
    model.eval()
    for i in NUSCENES_VISUALIZATION_IDXS:
        data = dataloader.dataset[i]
        data = torch_geometric.data.Batch.from_data_list([data])
        curve_idxs_orig = data.curve_idxs.clone()
        data = data.to(device)
        out = model(data, labels=data.y)
        _, pnt_errs = seg_loss(out, data.y, use_lovasz=use_lovasz)

        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for j, (pos, pred, y, per_pnt_errs, curve_idxs) in enumerate(zip(data.pos.split(sizes), out.split(sizes), data.y.split(sizes), pnt_errs.split(sizes), curve_idxs_orig.split(sizes))):
            # step 1: filter out where GT is noise
            pos = pos[y != NUSCENES_IGNORE_LABEL]
            pred = pred[y != NUSCENES_IGNORE_LABEL]
            per_pnt_errs = per_pnt_errs[y != NUSCENES_IGNORE_LABEL]
            y = y[y != NUSCENES_IGNORE_LABEL]

            # understand scene accuracy
            pred_classes = pred.argmax(dim=-1)
            correct_mask = pred_classes == y
            visualize_nuscenes_seg_mitsuba(pos, pred_classes, y, per_pnt_errs, correct_mask, curve_idxs, idx=i)


def visualize_nuscenes_seg_mitsuba(pc, pred_labels, gt_labels, errs, correct_mask, curve_idxs, title="Nusc Segmentation Vis", idx=0):
    print("Visualizing %s" % idx)
    pc, pred_labels, gt_labels, errs , correct_mask= pc.to('cpu').data.numpy(), pred_labels.to('cpu').data.numpy(), gt_labels.to('cpu').data.numpy(), errs.to('cpu').data.numpy(), correct_mask.to('cpu').data.numpy()
    pc = pc * 20 + np.array([0, 0, 20])
    cmap = NUSCENES_CMAP
    cmap = np.array(cmap) / 256

    # prediction colors
    pred_seg_clrs = cmap[pred_labels]
    gt_seg_clrs = cmap[gt_labels]
    clrs_correct = np.ones((pc.shape[0], 3)) * 0.55
    clrs_correct[~correct_mask] = np.array([1.0, 0, 0])

    # render each image
    imgs = []
    img = render_pc_kitti(pc, pred_seg_clrs, point_radius=0.0025)
    imgs.append(np.array(img ** (1.0 / 2.2)))
    img = render_pc_kitti(pc, clrs_correct, point_radius=0.0025)
    imgs.append(np.array(img ** (1.0 / 2.2)))
    img = render_pc_kitti(pc, gt_seg_clrs, point_radius=0.0025)
    imgs.append(np.array(img ** (1.0 / 2.2)))

    # render curve idxs
    curve_reds = [float(hash(str(idx) + 'r') % 256) / 255 for idx in curve_idxs.tolist()]
    curve_greens = [float(hash(str(idx) + 'g') % 256) / 255 for idx in curve_idxs.tolist()]
    curve_blues = [float(hash(str(idx) + 'b') % 256) / 255 for idx in curve_idxs.tolist()]
    curve_clrs = np.stack([curve_reds, curve_greens, curve_blues], axis=1)
    img = render_pc_kitti(pc, curve_clrs, point_radius=0.0025)
    imgs.append(np.array(img ** (1.0 / 2.2)))

    # concat and save
    full_img = np.concatenate(imgs, axis=1)
    cv2.imwrite(title+"%s.png" % idx, np.clip(full_img * 255, 0, 255).astype(int)[:, :, [2, 1, 0]])


def seg_loss(pred, gt, ignore=0, use_lovasz=False, ce_weights=None):
    pred_log_soft = F.log_softmax(pred, dim=-1)
    loss = F.nll_loss(pred_log_soft, gt, reduction='mean', ignore_index=ignore, weight=ce_weights)
    perpnt_loss = F.nll_loss(pred_log_soft, gt, reduction="none")

    # compute lovasz
    if use_lovasz:
        pred_soft = F.softmax(pred, dim=-1)
        pred_soft_filtered = pred_soft[gt != ignore]
        gt_filtered = gt[gt != ignore]
        lovasz_loss = lovasz_softmax_flat(pred_soft_filtered, gt_filtered)
        loss += torch.mean(lovasz_loss)

    return loss, perpnt_loss


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)  # +1 for indexing, then another +1 for class 0 not being used!
    hist = hist[unique_label + 1, :]  # unique_label+1 is aranging 1 to 20
    hist = hist[:, unique_label + 1]
    return hist
