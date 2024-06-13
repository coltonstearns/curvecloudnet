import torch
import time
import wandb
import tqdm
import numpy as np
import torch.nn.functional as F
import gc
import cv2

from src.visualization.visualize_plotly import visualize_seg
from torch_geometric.nn import global_mean_pool
from src.models.utils.lovasz_losses import lovasz_softmax_flat
from src.utils.utils import fast_hist, per_class_iu
from src.visualization.mitsuba_render import render_pc_audi
from src.run.globals import AUDI_VISUALIZE_IDXS, AUDI_IGNORE_LABEL, AUDI_CMAP, LOGGED_ERRORS


def train(model, dataloader, optimizer, device, start_idx, use_lovasz=False, weighted_ce=False):
    model.train()
    print_loss = 0
    t_init = time.time()
    all_point_errors = torch.zeros(0).float().to('cpu')
    end_idx = len(dataloader) - start_idx
    optimizer.zero_grad()
    for i, data in enumerate(dataloader):
        data = data.to(device)
        try:
            out = model(data, labels=data.y)
            loss, errs = seg_loss_audi(out, data.y, ignore=AUDI_IGNORE_LABEL, use_lovasz=use_lovasz)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        except RuntimeError as e:
            print("Error: ")
            print(e)
            torch.cuda.empty_cache()
            gc.collect()
            return False, i + start_idx

        # get errors and record
        point_errs = global_mean_pool(errs.view(-1, 1), batch=data.batch).cpu()
        all_point_errors = torch.cat([all_point_errors, point_errs])
        print_loss += loss.item()

        if (i + 1) % 10 == 0:
            avg_point_err = torch.mean(all_point_errors[max(0,(i-10)*dataloader.batch_size):i*dataloader.batch_size]).item()
            print(f'[{i+1+start_idx}/{len(dataloader)}] Loss: {print_loss / 4:.4f} '
                  f'Train NLL Error: {avg_point_err:.4f}')
            print("Total Time: %s" % (time.time() - t_init))
            wandb.log({"Train-Loss": print_loss, "Train-Error": avg_point_err})
            print_loss = 0
            t_init = time.time()

        if i > end_idx:
            break

    return True, 0


@torch.no_grad()
def val(model, dataloader, device, use_lovasz=False, test_mode=False, outdir=None):
    print("Evaluating...")
    model.eval()
    total_loss = 0

    # get class mappings
    unique_label_str = dataloader.dataset.config['learning_map_inv_names']
    unique_label_str = [unique_label_str[k] for k in sorted(unique_label_str.keys())][:-1]
    unique_label = np.arange(len(unique_label_str))  # 0 to 11, for 13 classes (where class 13th is "other")

    # set up profiling
    flops, times, mem_use = [], [], []
    ious, categories, histogram_counts = [], [], []
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # iterate through dataloader
    for i, data in enumerate(tqdm.tqdm(dataloader)):
        data = data.to(device)
        t0 = time.time()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        out = model(data, labels=data.y)
        t1 = time.time()
        times.append(t1 - t0)
        mem_use.append(torch.cuda.max_memory_allocated() / 10**9)
        out = F.log_softmax(out, dim=-1)

        # compute validation loss
        loss, _ = seg_loss_audi(out, data.y, ignore=AUDI_IGNORE_LABEL, use_lovasz=use_lovasz)
        if len(loss.size()) > 0:
            total_loss = total_loss + loss.item() if loss.size(0) > 0 else total_loss  # loss is sizeless in test split

        # compute histogram of per-class errors
        predict_labels = torch.argmax(out, dim=-1)
        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for pred, y in zip(predict_labels.split(sizes), data.y.split(sizes)):
            counts_hist = fast_hist_crop(pred.cpu().data.numpy(), y.cpu().data.numpy(), unique_label)
            histogram_counts.append(counts_hist)

    # compute per-class IOU
    per_class_iou = per_class_iu(sum(histogram_counts))
    miou = np.nanmean(per_class_iou)

    # record statistics
    mean_flops = np.mean(flops)
    mean_runtime = np.mean(times[5:])
    memuse = np.mean(mem_use[5:])

    # print statistics
    print("========== Validation Results ===========")
    print("Class IOU:")
    print(unique_label_str)
    print(per_class_iou)
    print(mem_use)
    print("mIOU: %s" % miou)
    print("=========================================")

    # Log wandb values
    LOGGED_ERRORS.append(miou)
    median_err_topk = np.median(sorted(LOGGED_ERRORS)[-5:])
    class_ious = {unique_label_str[i]: per_class_iou[i] for i in range(len(unique_label_str))}
    wandb.log({"mIOU": miou, "FLOPS": mean_flops, "Loss": total_loss,
               "Median Top 5 IOU": median_err_topk, "Runtime": mean_runtime, "GPU Memory Allocated": mem_use})
    wandb.log(class_ious)
    return miou


@torch.no_grad()
def viz_seg(model, dataloader, device, num_samples, use_lovasz=False, is_kortx=False):
    print("Visualizing...")
    model.eval()
    for i, data in enumerate(tqdm.tqdm(dataloader)):
        data = data.to(device)
        out = model(data, labels=data.y)
        _, pnt_errs = seg_loss_audi(out, data.y, ignore=AUDI_IGNORE_LABEL, use_lovasz=use_lovasz)

        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for j, (pos, pred, y, per_pnt_errs) in enumerate(zip(data.pos.split(sizes), out.split(sizes), data.y.split(sizes), pnt_errs.split(sizes))):
            if i not in AUDI_VISUALIZE_IDXS:
                continue

            # quick plotly visualization
            pred_classes = pred.argmax(dim=-1)
            correct_mask = pred_classes == y
            visualize_seg(pos, pred_classes, y, per_pnt_errs, correct_mask, max_classes=20, range=30)

            # very slow, but nicer looking visualization
            visualize_audi_seg_mitsuba(pos, pred_classes, y, correct_mask, idx=i)


def visualize_audi_seg_mitsuba(pc, pred_labels, gt_labels, correct_mask, title="Segmentation Vis", idx=0):
    pc, pred_labels = pc.to('cpu').data.numpy(), pred_labels.to('cpu').data.numpy()
    gt_labels, correct_mask = gt_labels.to('cpu').data.numpy(), correct_mask.to('cpu').data.numpy()
    pc *= 25

    # prediction colors
    pred_seg_clrs = AUDI_CMAP[pred_labels]
    gt_seg_clrs = AUDI_CMAP[gt_labels]

    # correct-prediction colors
    clrs_correct = np.ones((pc.shape[0], 3)) * 0.55
    clrs_correct[~correct_mask] = np.array([1.0, 0, 0])

    # render prediction, prediction errors, and gt segmentation
    imgs = []
    print("Starting rendering... this is usually quite slow.")
    img = render_pc_audi(pc, pred_seg_clrs, point_radius=0.0025)
    imgs.append(np.array(img ** (1.0 / 2.2)))
    img = render_pc_audi(pc, clrs_correct, point_radius=0.0025)
    imgs.append(np.array(img ** (1.0 / 2.2)))
    img = render_pc_audi(pc, gt_seg_clrs, point_radius=0.0025)
    imgs.append(np.array(img ** (1.0 / 2.2)))
    full_img = np.concatenate(imgs, axis=1)
    cv2.imwrite(title+"%03d.png" % idx, np.clip(full_img * 255, 0, 255).astype(int)[:, :, [2, 1, 0]])


def seg_loss_audi(pred, gt, ignore=0, use_lovasz=False, ce_weights=None):
    pred_log_soft = F.log_softmax(pred, dim=-1)
    loss = F.nll_loss(pred_log_soft, gt, reduction='mean', ignore_index=ignore, weight=ce_weights)
    perpnt_loss = F.nll_loss(pred_log_soft, gt, reduction="none")

    # compute lovasz
    if isinstance(use_lovasz, bool):
        if use_lovasz:
            pred_soft = F.softmax(pred, dim=-1)
            pred_soft_filtered = pred_soft[gt != ignore]
            gt_filtered = gt[gt != ignore]
            lovasz_loss = lovasz_softmax_flat(pred_soft_filtered, gt_filtered)
            loss += torch.mean(lovasz_loss)
    elif isinstance(use_lovasz, str):
        if use_lovasz == "lovasz-only":
            pred_soft = F.softmax(pred, dim=-1)
            pred_soft_filtered = pred_soft[gt != ignore]
            gt_filtered = gt[gt != ignore]
            lovasz_loss = lovasz_softmax_flat(pred_soft_filtered, gt_filtered)
            loss = torch.mean(lovasz_loss)

    return loss, perpnt_loss


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)  # +1 for indexing, then another +1 for class 16 not being used!
    hist = hist[unique_label, :]  # only take the first 12!
    hist = hist[:, unique_label]
    return hist