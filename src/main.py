import os.path as osp
import os
import json
import wandb
import numpy as np
import torch
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = "python"

from torch_geometric.loader import DataLoader as DataLoaderTorchGeo
from src.utils.config_utils import get_argparse_input
from src.utils.run_utils import select_task
from src.utils.load_utils import load_model, load_dataset, load_scheduler


def main(config):
    # Set up wandb
    wandb.init(project="CurveCloudNet", config=config, resume='allow')
    wandb.define_metric("mIOU", summary="max")
    wandb.define_metric("inst-mIOU", summary="max")
    wandb.define_metric("Class Accuracy", summary="max")
    wandb.define_metric("Mean F1", summary="max")

    # Uncomment if we plan want to preempt runs
    # wandb.mark_preempting()

    # set up dataset
    outdir = config['outdir']
    train_dataset, val_dataset, test_dataset, outdir = load_dataset(config, outdir)

    # set up dataloaders
    train_loader = DataLoaderTorchGeo(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0, drop_last=True)
    val_batch_size = 1 if config['dataset_source'] in ["kitti", "nuscenes"] else config['batch_size']
    val_loader = DataLoaderTorchGeo(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoaderTorchGeo(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=0) if test_dataset is not None else None

    # Create output directory
    if "sweep" in outdir.lower():
        outdir = os.path.join(outdir, wandb.run.id)
    if not osp.exists(outdir):
        os.makedirs(outdir)

    # determine which task we are running
    train, val, viz, out_dim = select_task(config['dataset_source'], config['task'], config['only_viz'])

    # set up model
    if 'use_additional_losses' in config:
        config['model']['use_additional_losses'] = config['use_additional_losses']
    config['model']['use_ce_weights'] = config['use_ce_weights']
    in_dim = train_dataset.in_dim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(config, in_dim, out_dim, device, outdir)
    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.log({"Model Parameters": model_total_params})

    # set up optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = load_scheduler(config, optimizer)
    momentum_decay = 1.0 if 'bn_momentum_decay' not in config else config['bn_momentum_decay']

    # check for preempted run state - occurs if respawning a failed run
    preempted_run_state = None
    if osp.exists(osp.join(outdir, "latest_state.json")):
        with open(osp.join(outdir, "latest_state.json"), "r") as f:
            preempted_run_state = json.load(f)

    # check if only validating or only visualizing
    if config['only_val']:
        val(model, val_loader, device, outdir=outdir)
        return
    if config['only_viz']:
        viz(model, val_loader if test_loader is None else test_loader, device, num_samples=10, is_kortx=test_loader is not None)
        return
    if 'only_test' in config and config['only_test']:
        val(model, test_loader, device, test_mode=True, outdir=outdir)
        return

    # enter training loop
    training_loop(train, val, viz, train_loader, val_loader, model, optimizer, scheduler, outdir,
                  momentum_decay, device, test_loader, preempted_run_state, config)


def training_loop(train, val, viz, train_loader, val_loader, model, optimizer, scheduler, outdir,
                  momentum_decay, device, test_loader=None, preempted_run_state=None, config=None):
    best_score = -np.inf if preempted_run_state is None else preempted_run_state['max_score']
    preempt_start_from = -1 if preempted_run_state is None else preempted_run_state['epoch']
    for epoch in range(config['epochs']):
        if epoch <= preempt_start_from:
            continue

        # run through training code
        print(">>>>> Epoch %s" % epoch)
        finished_training, train_idx = False, 0
        while not finished_training:
            finished_training, train_idx = train(model, train_loader, optimizer, device, train_idx, config['use_lovasz_loss'], config['use_ce_weights'])
        scheduler.step()

        # run through validation
        if (epoch + 1) % config['val_every'] == 0:
            # run on validation and (optionally) test set
            score = val(model, val_loader, device, config['use_lovasz_loss'], test_mode=False, outdir=outdir)
            if test_loader is not None:
                score = val(model, test_loader, device, config['use_lovasz_loss'], test_mode=False, outdir=outdir, prefix="Test ")

            # keep maximum score
            better_score = False
            if score > best_score:
                better_score = True
                best_score = score

            # save model
            if epoch % config['save_every'] == 0 or better_score:
                save_path = osp.join(outdir, "model_epoch%s.pth" % epoch)
                torch.save(model.state_dict(), save_path)
            if better_score:
                save_path = osp.join(outdir, "BEST_model_epoch%s.pth" % epoch)
                torch.save(model.state_dict(), save_path)

        # visualize (if applicable)
        if epoch % config['save_every'] == (config['save_every']-1) and viz is not None:
            viz(model, test_loader if test_loader is not None else val_loader, device, num_samples=3,
                use_lovasz=config['use_lovasz_loss'], is_kortx=test_loader is not None)

        # update batch-norm momentum
        new_bn_momentum = max(0.01, 0.1 * momentum_decay**epoch)
        model = model.apply(lambda x: bn_momentum_adjust(x, new_bn_momentum))

        # update preempting checkpoint (for if training code crashes)
        dump_preempting_checkpoint(model, optimizer, scheduler, outdir, epoch, best_score)


def dump_preempting_checkpoint(model, optimizer, scheduler, outdir, epoch, max_score):
    tmp_save_path = osp.join(outdir, "latest_model.pth")
    tmp_opt_save_path = osp.join(outdir, "latest_optimizer.pth")
    tmp_sched_save_path = osp.join(outdir, "latest_scheduler.pth")
    torch.save(model.state_dict(), tmp_save_path)
    torch.save(optimizer.state_dict(), tmp_opt_save_path)
    torch.save(scheduler.state_dict(), tmp_sched_save_path)

    other_state = {"epoch": epoch, "max_score": max_score}
    with open(osp.join(outdir, "latest_state.json"), "w") as f:
        json.dump(other_state, f)


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


if __name__ == "__main__":
    config = get_argparse_input()
    main(config)
