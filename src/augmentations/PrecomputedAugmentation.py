from ..datastructures import DataSample
from ..interfaces import AugmentationStrategy

class PrecomputedAugmentation(AugmentationStrategy):
    """ Loads a dataset produced by an out-of-band process to return data augmentation samples.

        This is particularly useful if data was augmented outside of the framework.
    Parameters:
        filepath (str): Path to tsv file to load. Expected format: path\tlabel
    """
    def __init__(self, filepath: str):
        self.filepath = filepath

    def augment_data(self, input_dataset):
        with open(self.filepath, 'r') as f:
            return [DataSample(*(x.split('\t'))) for x in f.readlines()]

