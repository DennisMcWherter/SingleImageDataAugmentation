from typing import List

import numpy as np

import torch
from torch.utils import data

from .datastructures import DataSample
from .image_utils import load_rgb_image
from .interfaces import Dataset

def convert_samples_to_dataset(samples: List[DataSample], transform=None, labelTransform=None) -> data.Dataset:
    return TorchDataset(samples, transform, labelTransform)

class TorchDataset(data.Dataset):
    """ Convert our internal representation of a dataset into a
        torch dataset containing labeled images.
    """

    def __init__(self, dataset: List[DataSample], transform=None, labelTransform=None):
        """ Construct a torch-compatible dataset
        Parameters:
            dataset (List[DataSample]): Input dataset
            transform: Transform to perform on dataset (default=None)
            labelTransform: Transform the label (default=None)
        """
        self.dataset = np.array(dataset)
        self.transform = transform
        self.labelTransform = labelTransform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.dataset[idx]
        image = load_rgb_image(sample.get_path())
        label = int(sample.get_label())

        item = (image, label)

        if self.transform:
            item = (self.transform(item[0]), item[1])

        if self.labelTransform:
            item = (item[0], self.labelTransform(item[1]))

        return item
