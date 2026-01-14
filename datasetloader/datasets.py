import os
import sys
import glob
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import math

class FluidflowDataset3D(Dataset):
    """
    Dataset class for the synthetic time-resolved 3D fluid flow dataset, named TR-Flow3D.
    Paper Reference: "comprising 19,500 training samples and 1,950 testing samples"
    """
    def __init__(self, npoints=2048, root='TR-Flow3D', partition='train'):
        # "Each sample consists of ten particle frames and nine corresponding ground-truth flow fields."
        self.nframes = 10
        
        # "Firstly, we directly sample a set of 2048 particles that are randomly seeded 
        # within the predefined three-dimensional observation volume"
        self.npoints = npoints 
        
        self.partition = partition
        self.root = root
        
        # Load data paths based on partition
        # "Ultimately, we generated a synthetic dataset comprising 19,500 training samples and 1,950 testing samples"
        if self.partition=='train':
            self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))
        else:
            self.datapath = glob.glob(os.path.join(self.root, 'TEST*.npz'))
        
        self.cache = {}
        self.cache_size = 30000

        self.datapath = [d for d in self.datapath]
        self.datapath.sort()

    def __getitem__(self, index):
        if index in self.cache:
            start_points, next_points_set, flow_set = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                # start_points represents frame 0
                start_points = data['start_points'].astype('float32')
                # next_points_set represents frames 1 to 9 (total 10 frames including start)
                next_points_set = data['next_points_set'].astype('float32')
                # "9 corresponding ground-truth flow fields"
                flow_set = data['flow_set'].astype('float32')

            if len(self.cache) < self.cache_size:
                self.cache[index] = (start_points, next_points_set, flow_set)

        if self.partition == 'train':
            # Create pairs of adjacent frames to simulate flow estimation between time steps
            # "corresponding flows between adjacent frames"
            times = np.arange(self.nframes)
            time_pairs = list(zip(times[:-1], times[1:])) # e.g., (0,1), (1,2)... (8,9)
            time_idx = np.random.choice(self.nframes-1, replace=False)
            
            # Combine start points and subsequent frames into a single sequence of shape (10, N, 3)
            pcs = np.concatenate([start_points[np.newaxis, ...], next_points_set], axis=0)
            
            # Select two adjacent frames (pos1, pos2) and the flow between them
            pos1 = pcs[time_pairs[time_idx][0]]
            pos2 = pcs[time_pairs[time_idx][1]]
            flow = flow_set[time_idx]

            # Randomly subsample points if the stored data has more points than self.npoints (2048)
            n1 = pos1.shape[0]
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            n2 = pos2.shape[0]
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

            pos1 = pos1[sample_idx1, :]
            pos2 = pos2[sample_idx2, :]
            flow = flow[sample_idx1, :]

            # Center the point cloud around the mean of the first frame
            # Note: The paper mentions "All particle sets are normalized to the cubic domain [0, 2pi]^3"
            # This step performs local centering for the network input.
            pos1_center = np.mean(pos1, 0)
            pos1 = pos1 - pos1_center
            pos2 = pos2 - pos1_center

            data_dict = {}
            # Input to the network: Concatenated source (pos1) and target (pos2) point sets
            data_dict['pcs'] = np.concatenate([pos1, pos2], axis=-1)
            data_dict['flow_3d'] = flow
        else:
            # Test partition: Load the full sequence
            pcs = np.concatenate([start_points[np.newaxis, ...], next_points_set], axis=0)
            pcs = pcs[:, :self.npoints, :]
            flow_set = flow_set[:, :self.npoints, :]

            data_dict = {}
            data_dict['pcs'] = pcs
            data_dict['flow_3d'] = flow_set
            data_dict['filenames'] = self.datapath[index]

        return data_dict

    def __len__(self):
        return len(self.datapath)

def build_train_dataset(dataset):
    if dataset == 'TR-Flow3D':
        # "TR-Flow3D... yielding 19,500 training samples"
        train_dataset = FluidflowDataset3D(npoints=2048, root='TR-Flow3D', partition='train')
    else:
        raise ValueError(f'stage {dataset} is not supported')

    return train_dataset

def build_test_dataset(dataset):
    if dataset == 'TR-Flow3D':
        # "TR-Flow3D... yielding ... 1,950 testing samples"
        test_dataset = FluidflowDataset3D(npoints=2048, root='TR-Flow3D', partition='test') 
    else:
        raise ValueError(f'stage {dataset} is not supported')

    return test_dataset