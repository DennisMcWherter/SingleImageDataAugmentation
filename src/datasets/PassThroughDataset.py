from ..interfaces import Dataset

class PassThroughDataset(Dataset):

    def get_training_data(self):
        return []

    def get_test_data(self):
        return []

    def get_holdout_data(self):
        return []

    def get_encoding(self):
        return {}
