"""Neonatal phase contrast image reconstruction dataset.

Defines the class and transforms.
"""

import os

import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils


class NeonatalPCDataset(Dataset):
    """Neonatal phase contrast images for image reconstruction.

    Args:
        data_dir: String. Root data directory. Should contain subfolders
            named 'train' and 'valid'.
        train: Boolean. For choosing training or validation data.
        transform: Optional transform to be applied on a sample.
    """
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        subfolder = 'train' if train is True else 'valid'
        datadir = os.path.join(data_dir, subfolder)
        self.files = [os.path.join(datadir, i) for i in os.listdir(datadir)]

    def __len__(self):
        """Enables using len() on the dataset."""
        return len(self.files)

    def __getitem__(self, idx):
        """Enables use of dataset[i].

        Loads the i-th sample from disk. Applies transforms if any.
        """
        data = sio.loadmat(self.files[idx])
        original = data['original']
        undersampled = data['undersampled']
        nx, ny, nt = original.shape
        nt = 24
        in_img = np.zeros((2, nx, ny, nt), dtype=np.float32)
        in_img[0] = original.real[:, :, 0:24]
        in_img[1] = original.imag[:, :, 0:24]
        out_img = np.zeros((2, nx, ny, nt), dtype=np.float32)
        out_img[0] = undersampled.real[:, :, 0:24]
        out_img[1] = undersampled.imag[:, :, 0:24]
        sample = [in_img, out_img]

        if self.transform:
            sample = self.transform(sample)

        return sample
