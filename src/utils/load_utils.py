import os.path as osp
import torch
import copy
import wandb
import gc

# datasets
from src.data.object_datasets import CurvesInMemoryDataset, SummerRoboticsDataset
from src.data.kitti_dataset import SemKITTI
from src.data.nuscenes_dataset import SemNuScenes
from src.data.audi_dataset import SemA2D2

# models
from src.models.base import ModelBase


def load_model(config, in_dim, out_dim, device, outdir):
    # build specified model
    config_cp = copy.deepcopy(config)
    model_kwargs = copy.deepcopy(config_cp['model'])
    assert(config_cp['model']['type'] == 'generic')  # this codebase does not support other baselines
    model_kwargs.pop('steps')
    model_kwargs.pop('feat_dims')
    model_kwargs.pop('out_mlp')
    model_kwargs['dataset_source'] = config_cp['dataset_source']
    model = ModelBase(in_dim, out_dim, steps=config_cp['model']['steps'], feat_dims=config_cp['model']['feat_dims'],
             out_mlp=config_cp['model']['out_mlp'], keypoint_visualization="none", **model_kwargs).to(device)

    # set weights if we have a checkpoint
    if not config['weights'] and osp.exists(osp.join(outdir, "latest_model.pth")):
        print("Models already exist in %s. Loading weights from latest checkpoint!" % outdir)
        config['weights'] = osp.join(outdir, "latest_model.pth")
        config['optimizer-weights'] = osp.join(outdir, "latest_optimizer.pth")
        config['scheduler-weights'] = osp.join(outdir, "latest_scheduler.pth")

    # load weights from file
    if config['weights']:
        print("Loading weights from %s" % config['weights'])
        state_dict = torch.load(config['weights'])
        model.load_state_dict(state_dict, strict=True)

    return model


def load_scheduler(config, optimizer):
    # build scheduler
    if 'scheduler' in config:
        if config['scheduler']['name'] == 'exp':
            lr_gamma = 1.0 if 'lr_gamma' not in config['scheduler'] else config['scheduler']['lr_gamma']
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
        elif config['scheduler']['name'] == 'cos-ann':
            eta_min = 0 if 'eta_min' not in config['scheduler'] else config['scheduler']['lr_gamma']
            T_0 = config['scheduler']['T_0']
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, eta_min=eta_min)
        else:
            raise RuntimeError("Not a valid schedule name")
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)

    # load scheduler and optimizer state, if available
    if 'optimizer-weights' in config:
        print("Loading Optimizer State!")
        optimizer_state = torch.load(config['optimizer-weights'])
        optimizer.load_state_dict(optimizer_state)
    if 'scheduler-weights' in config:
        print("Loading Scheduler State!")
        scheduler_state = torch.load(config['scheduler-weights'])
        scheduler.load_state_dict(scheduler_state)

    return scheduler


def load_dataset(config, outdir):
    # Kitti
    if config['dataset_source'] == "kitti":
        polarmix = False if "polarmix" not in config else config['polarmix']
        train_dataset = SemKITTI(config['data_path'], config['kitti_yaml_path'], split='train', polarmix_aug=polarmix)
        val_dataset = SemKITTI(config['data_path'],  config['kitti_yaml_path'], split='val')
        if 'only_test' in config and config['only_test']:
            test_dataset = SemKITTI(config['data_path'], config['kitti_yaml_path'], split='test')
        else:
            test_dataset = None

    # Audi
    elif config['dataset_source'] == 'audi':
        train_dataset = SemA2D2(config['data_path'], config['audi_yaml_path'], split='train')
        val_dataset = SemA2D2(config['data_path'], config['audi_yaml_path'], split='val')
        test_dataset = None

    # NuScenes
    elif config['dataset_source'] == 'nuscenes':
        from nuscenes import NuScenes
        if 'only_test' in config and config['only_test']:
            nusc_version = 'v1.0-test'
            nusc = NuScenes(version=nusc_version, dataroot=config['data_path'], verbose=True)
            test_dataset = SemNuScenes(config['data_path'], config['nuscenes_yaml_path'], nusc, split='test')
            val_dataset = test_dataset
            train_dataset = test_dataset
        else:
            nusc_version = 'v1.0-trainval'
            nusc = NuScenes(version=nusc_version, dataroot=config['data_path'], verbose=True)
            polarmix = False if "polarmix" not in config else config['polarmix']
            train_dataset = SemNuScenes(config['data_path'], config['nuscenes_yaml_path'], nusc, split='train', polarmix_aug=polarmix)
            val_dataset = SemNuScenes(config['data_path'], config['nuscenes_yaml_path'], nusc, split='val')
            test_dataset = None

    # Kortx
    elif config['dataset_source'] == 'kortx':
        line_density, npoints, resolution = config['data_generation']['line_density'], config['data_generation']['num_points'], config['data_generation']['resolution']
        laser_motion, dataset_source = config['data_generation']['laser_motion'], config['dataset_source']
        train_dataset = CurvesInMemoryDataset(config['data_path'], npoints, resolution, line_density, laser_motion, split='train', dataset_source=dataset_source,use_additional_losses=config['use_additional_losses'])
        val_dataset = CurvesInMemoryDataset(config['data_path'], npoints, resolution, line_density, laser_motion, split='val', dataset_source=dataset_source,use_additional_losses=config['use_additional_losses'])
        test_dataset = SummerRoboticsDataset(config['data_path'], npoints, dataset_source=dataset_source)
        wandb.define_metric("Test mIOU", summary="max")
        wandb.define_metric("Test inst-mIOU", summary="max")

    # ShapeNet
    else:
        line_density, npoints, resolution = config['data_generation']['line_density'], config['data_generation']['num_points'], config['data_generation']['resolution']
        laser_motion, dataset_source = config['data_generation']['laser_motion'], config['dataset_source']
        train_dataset = CurvesInMemoryDataset(config['data_path'], npoints, resolution, line_density, laser_motion, split='train', dataset_source=dataset_source,use_additional_losses=config['use_additional_losses'])
        val_dataset = CurvesInMemoryDataset(config['data_path'], npoints, resolution, line_density, laser_motion, split='val', dataset_source=dataset_source,use_additional_losses=config['use_additional_losses'])
        outdir = outdir + "{0}_{1}_{2}".format(npoints, resolution, line_density)
        test_dataset = None

    return train_dataset, val_dataset, test_dataset, outdir

