from abc import ABCMeta
from typing import List

from .datastructures import DataSample

class Dataset(metaclass=ABCMeta):
    """ Represents a full dataset.
        
        Datasets are represented as /paths/ on disk.
    """

    def prepare():
        """ Called before fetching data from the dataset.
            This method gives the class a chance to internally prepare
            for exported data
        """
        pass

    def get_training_data() -> List[DataSample]:
        """ Fetch a list of training data paths and labels

        Returns:
            List[DataSample]: List of data samples
        """
        pass

    def get_test_data() -> List[DataSample]:
        """ Fetch a list of test data paths and labels

        Returns:
            List[DataSample]: List of data samples
        """
        pass

    def get_holdout_data() -> List[DataSample]:
        """ Fetch a list of holdout data paths and labels

        Returns:
            List[DataSample]: List of data samples
        """
        pass

    def get_encoding():
        """ Get the one-hot encoding of categories to numeric values
        
        Returns:
            dict: From string values to numeric encoded values
        """
        pass

class RepresentativeSelection(metaclass=ABCMeta):
    """ Dataset representative selection algorithm.
    """

    def select_samples(input_dataset: List[DataSample]) -> List[DataSample]:
        """ Select a set of representative samples from the input data.

        Parameters:
            input_dataset (List[DataSample]): Full input dataset
        Returns:
            List[DataSample]: List of representative samples
        """
        pass

class AugmentationStrategy(metaclass=ABCMeta):
    """ Strategy for executing data augmentation.
    """

    def augment_data(dataset: List[DataSample], representatives: List[DataSample]) -> List[DataSample]:
        """ Augment a dataset.

        Parameters:
            List[DataSample]: Full input dataset to augment
            List[DataSample]: Selected representatives used in augmentation
        Returns:
            List[DataSample]: Augmented samples (i.e. not the entire augmented dataset). These samples will
                              automatically be concatenated to the existing training set.
        """
        pass

class TrainingStrategy(metaclass=ABCMeta):
    """ Implementation for specific model training.
    """

    def train(train_set: List[DataSample], test_set: List[DataSample]) -> str:
        """ Train a model based on input data.

        Parameters:
            train_set (List[DataSample]): Training dataset
            test_set (List[DataSample]): Test dataset
        Returns:
            str: Path to trained model
        """
        pass

class EvaluationStrategy(metaclass=ABCMeta):
    """ Model evaluation strategy.
    """

    def evaluate(holdout_set: List[DataSample], model_path: str) -> str:
        """ Load a model, evaluate against the holdout set, and store results.

        Parameters:
            holdout_set (List[DataSample]): Holdout dataset
            model_path (str): Path to load model from
        Returns:
            str: Path to stored results
        """
        pass

