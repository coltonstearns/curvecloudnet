import json
import os
import os.path as osp
import shutil

import torch

from torch_geometric.data import (Data)
from torch_geometric.io import read_txt_array
from torch_geometric.datasets import ShapeNet


class ShapeNetSegDataset(ShapeNet):

    def __init__(self, root, categories=None, split='trainval'):
        if categories == "all":
            categories = None
        super(ShapeNetSegDataset, self).__init__(root, categories, include_normals=False, split=split)

    def process_filenames(self, filenames):
        data_list = []
        categories_ids = [self.category_ids[cat] for cat in self.categories]
        cat_idx = {categories_ids[i]: i for i in range(len(categories_ids))}

        for name in filenames:
            cat = name.split(osp.sep)[0]
            if cat not in categories_ids:
                continue

            model_id = name.split(osp.sep)[1][:-4]
            synset_id = name.split(osp.sep)[0]

            data = read_txt_array(osp.join(self.raw_dir, name))
            pos = data[:, :3]
            x = data[:, 3:6]
            y = data[:, -1].type(torch.long)
            data = Data(pos=pos, x=x, y=y, category=cat_idx[cat], model_id=model_id, synset_id=synset_id)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        return data_list

    def process(self):
        trainval = []
        for i, split in enumerate(['train', 'val', 'test']):
            path = osp.join(self.raw_dir, 'train_test_split',
                            f'shuffled_{split}_file_list.json')
            with open(path, 'r') as f:
                filenames = [
                    osp.sep.join(name.split('/')[1:]) + '.txt'
                    for name in json.load(f)
                ]  # Removing first directory.
            data_list = self.process_filenames(filenames)
            if split == 'train' or split == 'val':
                trainval += data_list
            torch.save(self.collate(data_list), self.processed_paths[i])
        torch.save(self.collate(trainval), self.processed_paths[3])