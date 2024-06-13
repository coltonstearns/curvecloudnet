import numpy as np


# All
LOGGED_ERRORS = []

# Audi
AUDI_IGNORE_LABEL = 12
AUDI_VISUALIZE_IDXS = [1159, 902, 354, 1602, 8, 165, 780, 1377, 1093, 1381, 334, 1460, 1013, 1841, 1288, 1019, 285, 1007,
                    1102, 67, 355, 1603, 1107, 356, 1581, 645, 325, 1839, 1317, 1793, 1175, 1653, 800, 1178, 918, 896,
                    522, 1815, 1267, 992, 1834, 1699, 1739, 103, 1829, 1000, 528, 333, 1384, 1565, 838, 460, 1003, 560,
                    1655, 1593, 1558, 39, 996, 779, 1770, 1672, 455, 1108, 228, 157, 1838, 1360, 66, 1600, 1733, 897, 901,
                    586, 330, 1649, 1673, 1701, 1790, 1797]
AUDI_CMAP = [[20, 20, 20], [162, 122, 162], [184, 178, 109], [255, 20, 20], [250, 230, 4], [10, 138, 60], [20, 20, 251],
        [170, 14, 254], [227, 88, 33], [255, 205, 240], [0, 250, 250], [243, 195, 0], [255, 145, 52]]
AUDI_CMAP = np.array(AUDI_CMAP) / 256


# Kitti
KITTI_CLASSES = ['unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
           'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building',
           'fence', 'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign']
KITTI_CLASS_WEIGHTS = np.array([3.1557, 8.7029, 7.8281, 6.1354, 6.3161, 7.9937, 8.9704, 10.1922, 1.6155, 4.2187, 1.9385,
                          5.5455, 2.0198, 2.6261, 1.3212, 5.1102, 2.5492, 5.8585, 7.3929])
KITTI_CLASS_WEIGHTS = KITTI_CLASS_WEIGHTS / np.sum(KITTI_CLASS_WEIGHTS)
KITTI_CMAP = {
    'unlabeled': [250, 230, 4],
    'bicycle': [255, 20, 20],
    'bicyclist': [255, 20, 20],
    'pole': [227, 88, 33],
    'car': [20, 20, 251],
    'other-vehicle': [0, 250, 250],
    'motorcycle': [243, 195, 100],
    'motorcyclist': [243, 195, 100],
    'person': [29, 35, 90],
    'traffic-sign': [250, 170, 0],
    'trunk': [162, 122, 162],
    'truck': [162, 122, 162],
    'road': [20, 20, 20],
    'other-ground': [228, 188, 255],
    'parking': [200, 188, 255],
    'fence': [200, 150, 255],
    'sidewalk': [255, 205, 240],
    'terrain': [10, 138, 60],
    'building': [184, 178, 109],
    'vegetation': [163, 51, 58]
}
KITTI_CMAP = [KITTI_CMAP[cls] for cls in KITTI_CLASSES]
KITTI_VISUALIZE_IDXS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# nuScenes
NUSCENES_IGNORE_LABEL = 0
NUSCENES_VISUALIZATION_IDXS = [3535, 3531, 3528, 2687, 1846, 1473, 1474, 3529, 3530, 257, 1478, 157, 254, 1479, 3532,
                      1481, 1483, 3533, 1485, 3534, 4876, 1451, 4379, 1191, 1533, 4518, 1745, 1749, 3712, 1453,
                      3933, 3923, 4525, 4480, 4477, 2275, 4412, 1098, 5397, 4411, 1872, 5646, 1964, 2145, 1873,
                      5671, 4493, 5967, 5167, 5206, 4191, 4874, 4557, 4859, 4858, 346, 2140, 5026, 1969, 5956, 594,
                      2868, 5827, 5825, 1769, 1267, 2191, 5306, 1060, 3733, 3734, 5369, 5377, 2866, 1464, 5793, 2857,
                      5406, 4416, 4417, 1024, 1054, 3134, 1007, 3140, 1074, 1006, 1003, 1062, 1061, 988, 2900, 3128,
                      1027, 1045, 1044, 854, 1034, 1039, 855, 5836, 1142, 1141, 5877, 5880, 5875, 3610, 2566, 2607,
                      5837, 4023, 5878, 5879, 1135, 791, 5894, 5888, 5891, 2613, 2588]
NUSCENES_CLASSES = ['noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian',
           'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation']
NUSCENES_CLASS_WEIGHTS = [0, 0.05413265, 0.1541931 , 0.06454133, 0.03805366, 0.08522725, 0.11693198, 0.07665045, 0.10206691,
                 0.06352202, 0.04739051, 0.02240727, 0.05527531, 0.03266324, 0.03265252, 0.02587772, 0.02841408]
NUSCENES_CMAP = {
    'noise': [250, 230, 4],
    'barrier': [190, 255, 0],
    'bicycle': [255, 20, 20],
    'bus': [227, 88, 33],
    'car': [20, 20, 251],
    'construction_vehicle': [0, 250, 250],
    'motorcycle': [243, 195, 100],
    'pedestrian': [29, 35, 90],
    'traffic_cone': [250, 170, 0],
    'trailer': [162, 122, 162],
    'truck': [162, 122, 162],
    'driveable_surface': [20, 20, 20],
    'other_flat': [228, 188, 255],
    'sidewalk': [255, 205, 240],
    'terrain': [10, 138, 60],
    'manmade': [184, 178, 109],
    'vegetation': [163, 51, 58]
}
NUSCENES_CMAP = [NUSCENES_CMAP[cls] for cls in NUSCENES_CMAP]


# ShapeNet
SHAPENET_CATEGORY_NAMES = ['Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife', 'Lamp', 'Laptop',
            'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table']
NUM_SHAPENET_SEG_CLASSES = 50
