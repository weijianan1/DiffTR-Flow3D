#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
# import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import math


class FluidflowDataset3D(Dataset):
    def __init__(self, npoints=2048, root='data_preprocessing/data_processed_maxcut_35_both_mask_20k_2k', partition='train'):
        self.nframes = 10
        self.npoints = npoints
        self.partition = partition
        self.root = root
        if self.partition=='train':
            self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, 'TEST*.npz'))
            # self.datapath = glob.glob(os.path.join(self.root, '*.npz')) # mhd1024 transition_bl channel uniform beltrami isotropic1024coarse
        self.cache = {}
        self.cache_size = 30000

        ###### deal with one bad datapoint with nan value
        self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d]
        ######
        self.datapath.sort()

    def __getitem__(self, index):
        if index in self.cache:
            start_points, next_points_set, flow_set = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                start_points = data['start_points'].astype('float32')
                next_points_set = data['next_points_set'].astype('float32')
                flow_set = data['flow_set'].astype('float32')
                particle_mask = data['particle_mask'].astype('float32')

            if len(self.cache) < self.cache_size:
                self.cache[index] = (start_points, next_points_set, flow_set)

        if self.partition == 'train':
            times = np.arange(self.nframes)
            time_pairs = list(zip(times[:-1], times[1:]))
            time_idx = np.random.choice(self.nframes-1, replace=False)
            pcs = np.concatenate([start_points[np.newaxis, ...], next_points_set], axis=0)
            pos1 = pcs[time_pairs[time_idx][0]]
            pos2 = pcs[time_pairs[time_idx][1]]
            flow = flow_set[time_idx]

            n1 = pos1.shape[0]
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            n2 = pos2.shape[0]
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

            pos1 = pos1[sample_idx1, :]
            pos2 = pos2[sample_idx2, :]
            flow = flow[sample_idx1, :]

            pos1_center = np.mean(pos1, 0)
            pos1 = pos1 - pos1_center
            pos2 = pos2 - pos1_center

            data_dict = {}
            data_dict['pcs'] = np.concatenate([pos1, pos2], axis=-1)
            data_dict['flow_3d'] = flow
        else:
            pcs = np.concatenate([start_points[np.newaxis, ...], next_points_set], axis=0)
            pcs = pcs[:, :self.npoints, :]
            flow_set = flow_set[:, :self.npoints, :]

            data_dict = {}
            data_dict['pcs'] = pcs
            data_dict['flow_3d'] = flow_set
            data_dict['filenames'] = self.datapath[index]
            data_dict['particle_mask'] = particle_mask

        return data_dict

    def __len__(self):
        return len(self.datapath)



