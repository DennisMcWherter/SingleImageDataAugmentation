import logging

import numpy as np

from ..interfaces import Dataset

logger = logging.getLogger(__name__)

class ImbalancedDataset(Dataset):

    def __init__(self, dataset: Dataset, min_pct_imbalance=0.2, max_pct_imbalance=0.9):
        """ Method to force imbalance on a dataset

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
