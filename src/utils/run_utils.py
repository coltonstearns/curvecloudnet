import src.run.shapenet_seg as shapenet_seg
import src.run.kitti_seg as kitti_seg
import src.run.audi_seg as audi_seg
import src.run.nuscenes_seg as nuscenes_seg
import src.run.shapenet_classification as shapenet_classification

# number of prediction classes for each dataset
NUM_SHAPENET_CLASSES = 50
NUM_SHAPENET_OBJECTS_TYPES = 16
NUM_KITTI_CLASSES = 20
NUM_AUDI_CLASSES = 13
NUM_NUSCENES_CLASSES = 17


def select_task(dataset_source, task, only_viz):
    if (dataset_source in ['shapenet-seg', 'kortx']) and task == 'segmentation':
        train = shapenet_seg.train
        val = shapenet_seg.val
        viz = shapenet_seg.viz_seg
        out_dim = NUM_SHAPENET_CLASSES
    elif (dataset_source in ['shapenet-seg', 'kortx']) and task == 'classification':
        train = shapenet_classification.train
        val = shapenet_classification.val
        viz = shapenet_classification.viz if only_viz else None
        out_dim = NUM_SHAPENET_OBJECTS_TYPES
    elif dataset_source == 'kitti' and task == 'segmentation':
        train = kitti_seg.train
        val = kitti_seg.val
        viz = None if not only_viz else kitti_seg.viz_seg
        out_dim = NUM_KITTI_CLASSES
    elif dataset_source == 'audi' and task == 'segmentation':
        train = audi_seg.train
        val = audi_seg.val
        viz = None if not only_viz else audi_seg.viz_seg
        out_dim = NUM_AUDI_CLASSES
    elif dataset_source == 'nuscenes' and task == 'segmentation':
        train = nuscenes_seg.train
        val = nuscenes_seg.val
        viz = None if not only_viz else nuscenes_seg.viz_seg
        out_dim = NUM_NUSCENES_CLASSES  # 1-16 are valid classes; class 0 is "other"
    else:
        raise RuntimeError()

    return train, val, viz, out_dim