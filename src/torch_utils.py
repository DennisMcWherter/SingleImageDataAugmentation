import logging
from typing import List, Tuple

import numpy as np

import torch

from .datastructures import DataSample
from .image_utils import load_rgb_image
from .interfaces import Dataset

logger = logging.getLogger(__name__)

def convert_samples_to_dataset(samples: List[DataSample], transform=None, labelTransform=None) -> torch.utils.data.Dataset:
    return TorchDataset(samples, transform, labelTransform)

def compute_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    out_array = outputs.data.cpu().detach().numpy()
    labels_array = labels.data.cpu().detach().numpy()
    results = out_array.argmax(axis=1)
    matches = results == labels_array
    return float(np.count_nonzero(matches)) / len(labels)

def test_model(model, loss_fn, test_inputs):
    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0.0
    for i, (inputs, labels) in enumerate(test_inputs):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        model.eval()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        accuracy = compute_accuracy(outputs, labels)

        if i % 5 == 0:
            logger.info('Minibatch test loss: {}, test accuracy: {}'.format(loss.item(), accuracy))

        total_loss += loss.item()
        total_accuracy += accuracy
        total_batches += 1.0

    avg_loss = total_loss / total_batches
    avg_accuracy = total_accuracy / total_batches

    return (avg_loss, avg_accuracy)

class TorchDataset(torch.utils.data.Dataset):
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
