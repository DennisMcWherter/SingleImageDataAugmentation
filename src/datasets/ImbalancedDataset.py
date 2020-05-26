import ..interfaces Dataset

class ImbalancedDataset(Dataset):

    def __init__(self, dataset: Dataset):
        """ Method to force imbalance on a dataset

        Parameters:
            dataset (Dataset): Dataset to cause imbalance
        """
        pass

    def get_training_data(self):
        pass

    def get_test_data(self):
        pass

    def get_holdout_data(self):
        pass
