import torch
import time
import wandb
import tqdm
import torchmetrics
import numpy as np
import torch.nn.functional as F
import plotly.express as px
import pandas as pd
import sklearn.metrics as metrics

from src.run.globals import SHAPENET_CATEGORY_NAMES


def train(model, dataloader, optimizer, device, start_idx, use_lovasz=False, use_ce_weights=False):
    model.train()

    print_loss = 0
    t0 = time.time()
    confmats = torch.zeros((0, len(SHAPENET_CATEGORY_NAMES), len(SHAPENET_CATEGORY_NAMES))).to(device)
    for i, data in enumerate(dataloader):
        # load and format data
        data = data.to(device)
        class_labels = data.labels.clone()
        delattr(data, "labels")

        # forward and backward passes
        optimizer.zero_grad()
        out = model(data)
        out = out.log_softmax(dim=-1)
        loss = F.nll_loss(out, class_labels)
        loss.backward()
        optimizer.step()

        # get errors and record
        confmat_compute = torchmetrics.ConfusionMatrix(num_classes=model.n_out).to(device)
        confmat = confmat_compute(np.e**out, class_labels)  # 10 x 10 matrix
        confmats = torch.cat([confmats, confmat.float().unsqueeze(0)], dim=0)
        print_loss += loss.item()

        if (i + 1) % 10 == 0:
            avg_conf_mat = torch.sum(confmats[i-10:i], dim=0)
            acc = confmat2acc(avg_conf_mat)
            print(f'[{i+1}/{len(dataloader)}] Loss: {print_loss / 10:.4f} '
                  f'Train Acc: {acc:.4f}')
            print("Total Time: %s" % (time.time() - t0))
            print_loss = 0
            t0 = time.time()
    return True, None


@torch.no_grad()
def val(model, dataloader, device, use_lovasz=False, test_mode=False, outdir=None, prefix=""):
    print("Evaluating...")
    model.eval()

    total_confmat = torch.zeros(len(SHAPENET_CATEGORY_NAMES), len(SHAPENET_CATEGORY_NAMES)).to(device)
    total_loss = 0
    gt_labels = []
    pred_labels = []
    flops, times, mem_use = [], [], []
    for i, data in enumerate(tqdm.tqdm(dataloader)):
        data = data.to(device)
        class_labels = data.labels.clone()
        delattr(data, "labels")

        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        t0 = time.time()
        out = model(data)
        out = out.log_softmax(dim=-1)
        t1 = time.time()
        times.append(t1 - t0)
        mem_use.append(torch.cuda.max_memory_allocated() / 10**9)

        # get errors and record
        loss = F.nll_loss(out, class_labels)
        total_loss += loss.item()

        # log confusion matrix
        confmat_compute = torchmetrics.ConfusionMatrix(num_classes=model.n_out).to(device)
        confmat = confmat_compute(np.e**out, class_labels)  # 10 x 10 matrix
        total_confmat += confmat

        # log labels
        gt_labels.append(class_labels)
        pred_labels.append(out.argmax(dim=-1))

    # get sklearn metrics
    gt_labels = torch.cat(gt_labels).cpu().numpy()
    pred_labels = torch.cat(pred_labels).cpu().numpy()
    acc = float("%.3f" % (100. * metrics.accuracy_score(gt_labels, pred_labels)))
    acc_avg = float("%.3f" % (100. * metrics.balanced_accuracy_score(gt_labels, pred_labels)))

    # summarize and record statistics
    avg_confmat = total_confmat.cpu().numpy() / torch.sum(total_confmat).item()
    avg_confmat = pd.DataFrame(data=avg_confmat, index=np.array(SHAPENET_CATEGORY_NAMES), columns=np.array(SHAPENET_CATEGORY_NAMES))
    fig = px.imshow(avg_confmat, text_auto=True)
    fig.update_layout(xaxis={'title': 'Predicted'}, yaxis={'title': 'Label'})

    # get precision-recall info
    precision, recall, f1, mean_f1 = confmat2precrec(total_confmat)
    logging = {SHAPENET_CATEGORY_NAMES[i] + "_precision": precision[i] for i in range(len(SHAPENET_CATEGORY_NAMES))}
    logging = {**logging, **{SHAPENET_CATEGORY_NAMES[i] + "_recall": recall[i] for i in range(len(SHAPENET_CATEGORY_NAMES))}}
    logging = {**logging, **{SHAPENET_CATEGORY_NAMES[i] + "_f1": f1[i] for i in range(len(SHAPENET_CATEGORY_NAMES))}}
    logging['Mean F1'] = mean_f1
    logging['Accuracy'] = acc
    logging['Class Accuracy'] = acc_avg
    logging['Confusion Matrix'] = fig

    # add in runtime stats
    mean_runtime = np.mean(times[2:])
    memuse = np.mean(mem_use[2:])
    logging['Runtime'] = mean_runtime
    logging['GPU Memory Allocated'] = memuse
    wandb.log(logging)

    print("========== Validation Results ===========")
    print("Mean Accuracy: %s,      " % (acc))
    print("Class Mean Accuracy: %s,      " % (acc_avg))
    print("Mean F1: %s,      " % (mean_f1))
    print("=========================================")

    return acc_avg


@torch.no_grad()
def viz(model, dataloader, device, num_samples, use_lovasz=False, is_kortx=False):
    return


def confmat2acc(confmat):
    eye = torch.eye(confmat.size(0)).to(confmat)
    return torch.sum(confmat * eye) / torch.sum(confmat)


def confmat2precrec(confmat):
    tp = torch.diagonal(confmat, 0)

    # compute false positives and false negatives per-class
    inv_eye = (torch.ones(confmat.size()) - torch.eye(confmat.size(0))).to(confmat)
    fp = (confmat * inv_eye).sum(dim=0)
    fn = (confmat * inv_eye).sum(dim=1)

    # compute false negatives per class
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    f1 = torch.nan_to_num(f1, nan=0)
    mean_f1 = f1.mean()
    return precision.cpu().numpy(), recall.cpu().numpy(), f1.cpu().numpy(), mean_f1.cpu().item()

