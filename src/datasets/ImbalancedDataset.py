import logging

import numpy as np

from ..interfaces import Dataset

logger = logging.getLogger(__name__)

class StratifiedRandomImbalancedDataset(Dataset):

    def __init__(self, dataset: Dataset, min_pct_imbalance=0.2, max_pct_imbalance=0.9, num_labels=3, labels=None):
        """ Method for randomly sampling within only specified labels. Imbalance is stratified by label.

        Parameters:
            dataset (Dataset): Dataset to induce imbalance in
            min_pct_imbalance (float): Minimum percentage to imbalance any class (default=0.2)
            max_pct_imbalance (float): Maximum percentage to imbalance any class (default=0.9)
            num_labels (int): Number of labels to induce imbalance (default=3)
            labels (List[int]): List of labels to specifically imbalance (default=None)
        """
        self.imbalanced = ImbalancedDataset(dataset, min_pct_imbalance, max_pct_imbalance)
        self.labels = labels
        self.num_labels = 3
        self.training_data = None
        self.test = None

    def prepare(self):
        # This will stratify and induce imbalance across all labels
        self.imbalanced.prepare()

        # Determine which labels should have imbalance
        labels = list(set([x.get_label() for x in self.imbalanced.get_training_data()]))
        np.random.shuffle(labels)
        selected_labels = set(self.labels if self.labels else labels[:self.num_labels])

        # Build a new dataset where selected classes come from imbalance and others remain untouched
        self.training_data = []
        for label in labels:
            if label in selected_labels:
                self.training_data.extend([x for x in self.imbalanced.get_training_data() if x.get_label() == label])
            else:
                self.training_data.extend([x for x in self.imbalanced.dataset.get_training_data() if x.get_label() == label])

    def get_training_data(self):
        return self.training_data

    def get_test_data(self):
        return self.imbalanced.get_test_data()

    def get_holdout_data(self):
        return self.imbalanced.get_holdout_data()

    def get_encoding(self):
        return self.imbalanced.get_encoding()

class ImbalancedDataset(Dataset):

    def __init__(self, dataset: Dataset, min_pct_imbalance=0.2, max_pct_imbalance=0.9):
        """ Method to force imbalance on a dataset. This class induces imbalance on the full dataset.
        Imbalance is stratified per class label.

        Parameters:
            dataset (Dataset): Dataset to cause imbalance
            min_pct_imbalance (float): Minimum percentage to imbalance any class (default=0.2)
            max_pct_imbalance (float): Maximum percentage to imbalance any class (default=0.9)
        """
        assert(min_pct_imbalance <= max_pct_imbalance)
        self.dataset = dataset
        self.min_pct_imbalance = min_pct_imbalance
        self.max_pct_imbalance = max_pct_imbalance
        self.training_data = None

    def prepare(self):
        self.dataset.prepare()

        logger.debug('-- Imbalancing dataset...')
        self.training_data = self.__imbalance()

        logger.debug('---- Samples before imbalance: {}\n---- Samples after imbalance: {}'.format(len(self.dataset.get_training_data()), len(self.training_data)))

    def get_training_data(self):
        return self.training_data

    def get_test_data(self):
        return self.dataset.get_test_data()

    def get_holdout_data(self):
        return self.dataset.get_holdout_data()

    def get_encoding(self):
        return self.dataset.get_encoding()

    def __imbalance(self):
        full_training_data = self.dataset.get_training_data()
        partitioned = {}

        # Partition based on class
        for sample in full_training_data:
            label = sample.get_label()
            if not label in partitioned:
                partitioned[label] = []
            partitioned[label].append(sample)

        # Drop random samples from each class
        for label in partitioned.keys():
            data = partitioned[label]

            # Shuffle data randomly
            np.random.shuffle(data)

            # Drop some percentage
            pct_drop = np.random.uniform(low=self.min_pct_imbalance, high=self.max_pct_imbalance)
            num_drop = int(pct_drop * len(data))

            # Make sure we retain at least one training sample
            assert(num_drop < len(data))

            # Retain only selected samples
            partitioned[label] = data[num_drop:]

        return [x for label in partitioned.keys() for x in partitioned[label]]
