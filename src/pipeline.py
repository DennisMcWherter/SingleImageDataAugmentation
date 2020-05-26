import json
import logging
from typing import List
import os

from .datastructures import DataSample
from .interfaces import Dataset, RepresentativeSelection, AugmentationStrategy, TrainingStrategy, EvaluationStrategy

logger = logging.getLogger(__name__)

class Pipeline:

    def __init__(self,
                 pipeline_name: str,
                 dataset: Dataset,
                 selection_strategy: RepresentativeSelection,
                 augmentation_strategy: AugmentationStrategy,
                 training_strategy: TrainingStrategy,
                 evaluation_strategy: EvaluationStrategy,
                 save_intermediate: bool=True):
        """ Evaluation pipeline.

        Parameters:
            pipeline_name (str): Pipeline name
            dataset (Dataset): Complete dataset
            selection_strategy (RepresentativeSelection): Representative selection algorithm
            augmentation_strategy (AugmentationStrategy): Method to augment representatives
            training_strategy (TrainingStrategy): Train a model with augmented data
            evaluation_strategy (EvaluationStrategy): Evaluate a trained model
            save_intermediate (bool): Write intermediate state files
        """
        self.pipeline_name = pipeline_name
        self.dataset = dataset
        self.selection_strategy = selection_strategy
        self.augmentation_strategy = augmentation_strategy
        self.training_strategy = training_strategy
        self.evaluation_strategy = evaluation_strategy
        self.save_intermediate = save_intermediate

        self.loaded_datasets = None
        self.loaded_representatives = None
        self.loaded_augmented_data = None
        self.loaded_trained_model = None
        logger.info('Created pipeline: {}'.format(pipeline_name))

    def execute(self) -> str:
        """ Execute the pipeline.

        Returns:
            str: Output location for pipeline results
        """
        logger.info('Executing pipeline: {}'.format(self.pipeline_name))

        training_data, test_data, holdout_data = self.__load_or_compute_dataset()

        representatives = self.__load_or_select_representatives(training_data)

        augmented_data = self.__load_or_augment_data(training_data, representatives)

        trained_model_path = self.__load_or_train_model(augmented_data, test_data)

        result_path = self.evaluation_strategy.evaluate(holdout_data, trained_model_path)

        logger.info('Evaluating results...')
        self.__write_output_metadata(trained_model_path, result_path)

        return result_path

    def restore_pipeline(self, dataset: bool=True, representatives: bool=True, augmentation: bool=True, training: bool=False):
        """ Restore pipeline from a given state.

        Parameters:
            dataset (bool): Restore dataset (default=True)
            representatives (bool): Restore selected representatives (default=True)
            augmentation (bool): Restore augmented dataset (default=True)
            training (bool): Restore trained model (default=False)
        """
        if dataset:
            training_data = self.__load_intermediate('training_data.csv')
            test_data = self.__load_intermediate('test_data.csv')
            holdout_data = self.__load_intermediate('holdout_data.csv')
            self.loaded_datasets = (training_data, test_data, holdout_data)
            logger.info('Loaded split datasets.')
        if representatives:
            self.loaded_representatives = self.__load_intermediate('representatives.csv')
            logger.info('Loaded representatives.')
        if augmentation:
            self.loaded_augmented_data = self.__load_intermediate('augmented_data.csv')
            logger.info('Loaded pre-augmented dataset.')
        if training:
            self.loaded_trained_model = self.__load_output_metadata('model_path')
            logger.info('Loaded pre-trained model.')

    def __load_or_compute_dataset(self):
        if not self.loaded_datasets:
            logger.info('Loading and preparing dataset...')
            self.dataset.prepare()

            training_data = self.dataset.get_training_data()
            test_data = self.dataset.get_test_data()
            holdout_data = self.dataset.get_holdout_data()

            # Write intermediate dataset results
            self.__write_intermediate('training_data.csv', training_data)
            self.__write_intermediate('test_data.csv', test_data)
            self.__write_intermediate('holdout_data.csv', holdout_data)
        
            return (training_data, test_data, holdout_data)
        
        return self.loaded_datasets

    def __load_or_augment_data(self, training_data: List[DataSample], representatives: List[DataSample]) -> List[DataSample]:
        if not self.loaded_augmented_data:
            logger.info('Augmenting data...')
            augmented_samples = self.augmentation_strategy.augment_data(representatives)
            self.__write_intermediate('augmented_samples.csv', augmented_samples)
            augmented_data = augmented_samples + training_data
            self.__write_intermediate('augmented_data.csv', augmented_data)
            return augmented_data
        return self.loaded_augmented_data

    def __load_or_select_representatives(self, training_data: List[DataSample]) -> List[DataSample]:
        if not self.loaded_representatives:
            logger.info('Selecting representatives...')
            representatives = self.selection_strategy.select_samples(training_data)
            self.__write_intermediate('representatives.csv', representatives)
            return representatives
        return self.loaded_representatives

    def __load_or_train_model(self, augmented_data, test_data):
        if not self.loaded_trained_model:
            logger.info('Training model...')
            return self.training_strategy.train(augmented_data, test_data)
        return self.loaded_trained_model

    def __write_intermediate(self, filename: str, data: List[DataSample]):
        if not self.save_intermediate:
            return

        self.__make_intermediate_dir()

        path = self.__intermediate_path(filename)
        with open(path, 'w') as f:
            for datum in data:
                f.write('{}\t{}\n'.format(datum.get_path(), datum.get_label()))

        logger.debug('-- Wrote filename: {}'.format(path))

    def __load_intermediate(self, filename: str) -> List[DataSample]:
        path = self.__intermediate_path(filename)
        with open(path, 'r') as f:
            return [DataSample(*(x.split('\t')),) for x in f.readlines()]

    def __write_output_metadata(self, model_path: str, result_path: str):
        if not self.save_intermediate:
            return

        self.__make_intermediate_dir()

        path = self.__intermediate_path('result-metadata.json')
        data = {'model_path': model_path, 'result_path': result_path}
        with open(path, 'w') as f:
            json.dump(data, f)

        logger.debug('-- Wrote result metadata: {}'.format(path))

    def __load_output_metadata(self, field):
        path = self.__intermediate_path('result-metadata.json')
        with open(path, 'r') as f:
            return json.load(path)[field]

    def __make_intermediate_dir(self):
        directory = self.__save_dir()
        if not os.path.exists(directory):
            os.makedirs(directory)

    def __intermediate_path(self, filename: str) -> str:
        return os.path.join(self.__save_dir(), filename)

    def __save_dir(self) -> str:
        return os.path.join('intermediate', self.pipeline_name)

