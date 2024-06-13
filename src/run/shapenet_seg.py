import torch
import time
import wandb
import tqdm
import numpy as np
import gc
import cv2

import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torchmetrics.functional import jaccard_index
import torch_geometric
from torch_scatter import scatter

from src.visualization.visualize_plotly import visualize_seg
from src.visualization.visualize_mitsuba import visualize_seg_mitsuba
from src.run.globals import NUM_SHAPENET_SEG_CLASSES, LOGGED_ERRORS, SHAPENET_CATEGORY_NAMES


def train(model, dataloader, optimizer, device, start_idx, use_lovasz=False, weighted_ce=False):
    model.train()
    print_loss = 0
    t_init = time.time()
    end_idx = len(dataloader) - start_idx

    point_errors = torch.zeros(0).float().to(device)
    for i, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        try:
            out = model(data, labels=data.y)
            loss, errs = seg_loss(out, data.y)
            loss.backward()
            optimizer.step()
        except RuntimeError as e:
            print("Error: ")
            print(e)
            torch.cuda.empty_cache()
            gc.collect()
            return False, i + start_idx

        # get errors and record
        these_point_errs = global_mean_pool(errs.view(-1, 1), batch=data.batch)
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

        if i > end_idx:
            break

    return True, None


@torch.no_grad()
def val(model, dataloader, device, use_lovasz=False, test_mode=False, outdir=None, prefix=""):
    print("Evaluating...")
    model.eval()
    total_loss = 0
    y_map = torch.empty(NUM_SHAPENET_SEG_CLASSES, device=device).long()
    times, mem_use = [], []
    ious, categories = [], []
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    for i, data in enumerate(tqdm.tqdm(dataloader)):
        # run forward pass through model
        data = data.to(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        t0 = time.time()
        out = model(data, labels=data.y)
        out = F.log_softmax(out, dim=-1)
        t1 = time.time()
        times.append(t1 - t0)
        mem_use.append(torch.cuda.max_memory_allocated() / 10**9)

        # compute validation loss
        loss, _ = seg_loss(out, data.y)
        total_loss += loss.item()

        # go through each object and compute its category IOU
        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        batch_categories = data.labels
        for out, y, category in zip(out.split(sizes), data.y.split(sizes), batch_categories):
            category = SHAPENET_CATEGORY_NAMES[category]
            part = torch_geometric.datasets.ShapeNet.seg_classes[category]
            part = torch.tensor(part, device=device)
            y_map[part] = torch.arange(part.size(0), device=device)

            # compute an IoU score for this object instance
            iou = jaccard_index(out[:, part].argmax(dim=-1), y_map[y], num_classes=part.size(0), absent_score=1.0)
            ious.append(iou.item())

        categories.append(torch.tensor(batch_categories).to(out.device))  # data.category

    # compute summaries: this will be a the mean over each separate class
    iou = torch.tensor(ious, device=device)
    category = torch.cat(categories, dim=0)
    iou = scatter(iou, category, reduce='mean')  # Per-category IoU.
    inst_iou = torch.mean(torch.tensor(ious, device=device)).item()

    # record statistics
    mean_iou = torch.mean(iou).item()
    med_iou = torch.median(iou).item()
    mean_runtime = np.mean(times[2:])
    memuse = np.mean(mem_use[2:])

    print("========== Validation Results ===========")
    print("Class Mean IOU: %s,      " % (mean_iou * 14 / 6))
    print("Instance Mean IOU: %s" % (inst_iou))
    print("=========================================")

    # Log wandb values
    LOGGED_ERRORS.append(mean_iou)
    median_err_topk = np.median(sorted(LOGGED_ERRORS)[-5:])
    wandb.log({prefix+"mIOU": mean_iou, prefix+"inst-mIOU": inst_iou, prefix+"Median IOU": med_iou, prefix+"Loss": total_loss,
               prefix+"Median Top 5 IOU": median_err_topk, prefix+"Runtime": mean_runtime, prefix+"GPU Memory Allocated": memuse})

    # Log per-class values
    per_category_stats = {}
    for i, category_name in enumerate(SHAPENET_CATEGORY_NAMES):
        per_category_stats[prefix+category_name] = iou[i].item()
        if i+1 == len(iou):
            break
    wandb.log(per_category_stats)

    return mean_iou


@torch.no_grad()
def viz_seg(model, dataloader, device, num_samples, use_lovasz=False, is_kortx=False):
    print("Visualizing...")
    model.eval()
    num_visualized, num_pca_visualized = 0, 0
    for i, data in enumerate(tqdm.tqdm(dataloader)):
        data = data.to(device)
        out = model(data, labels=data.y)
        _, pnt_errs = seg_loss(out, data.y)

        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        y_map = torch.empty(NUM_SHAPENET_SEG_CLASSES, device=device).long()
        batch_categories = data.labels
        for j, (pos, out, y, per_pnt_errs, category) in enumerate(zip(data.pos.split(sizes), out.split(sizes), data.y.split(sizes), pnt_errs.split(sizes), batch_categories)):
            category = SHAPENET_CATEGORY_NAMES[category]
            part = torch_geometric.datasets.ShapeNet.seg_classes[category]
            part = torch.tensor(part, device=device)
            y_map[part] = torch.arange(part.size(0), device=device)
            pred_classes = out[:, part].argmax(dim=-1)
            correct_mask = pred_classes == y_map[y]

            # rotate 90 degrees in X-axis and 90 degrees in Z-axis
            if not is_kortx:
                rot_mat = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                pos = pos @ rot_mat.T.to(pos)
            # shapenet - transform into same space as Kortx
            else:
                R_x = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.double).cuda()
                R_x_2 = torch.tensor([[1, 0, 0], [0, np.cos(np.pi / 7), -np.sin(np.pi / 7)], [0, np.sin(np.pi / 7), np.cos(np.pi / 7)]], dtype=torch.double).cuda()
                pos /= 2
                pos = pos.cuda().double() @ R_x.T @ R_x_2.T + 0.5 + torch.tensor([[0, 0, 0.1]]).cuda()

            if num_visualized > num_samples:
                break

            # visualize plotly segmentation
            visualize_seg(pos, pred_classes, y_map[y], per_pnt_errs, correct_mask)
            if is_kortx:
                camera_origin = (3.0, 9.0 / 4.0, 9.0 / 4.0)
                visualize_seg_mitsuba(pos, pred_classes, y_map[y], per_pnt_errs, correct_mask, camera_origin=camera_origin, is_kortx=is_kortx, index=i)
            else:
                visualize_seg_mitsuba(pos, pred_classes, y_map[y], per_pnt_errs, correct_mask, index=i)
            num_visualized += 1


def seg_loss(pred, gt):
    pred = F.log_softmax(pred, dim=-1)
    perpnt_loss = F.nll_loss(pred, gt, reduction="none")
    loss = torch.mean(perpnt_loss)
    return loss, perpnt_loss
