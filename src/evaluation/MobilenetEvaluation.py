import logging
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from ..interfaces import EvaluationStrategy
from ..models.MobilenetV2 import TestMobilenetV2
from ..torch_utils import convert_samples_to_dataset, test_model

logger = logging.getLogger(__name__)

class MobilenetV2EvaluationStrategy(EvaluationStrategy):

    def __init__(self, output_path, num_classes):
        """ MobilenetV2 evaluation strategy.
        Parameters:
            output_path (str): Location where output results file is written
            num_classes (int): Number of classes for Mobilenet to classify
        """
        self.output_path = output_path
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()

    def evaluate(self, holdout_set, model_path):
        logger.info('Evaluating model at path: {}'.format(model_path))
        model = self.__load_model(model_path)
        dataset = self.__to_dataset(holdout_set)

        start = time.time()
        test_results = test_model(model, self.loss_fn, dataset)
        end = time.time()
        total_time = end - start

        logger.info('Done evaluating (took {} seconds)'.format(total_time))

        path = os.path.join(self.output_path, 'results.txt')

        results = (*test_results, total_time)
        result_str = 'Holdout Loss: {}\nHoldout Accuracy: {}\nEvaluation Time: {}\n'.format(*results)
        logger.info("Network Results\n----------------\n{}".format(result_str))

        self.__write_results(path, result_str)

        return path

    def __load_model(self, model_path):
        model = TestMobilenetV2(num_classes=self.num_classes)
        model.load_state_dict(torch.load(model_path))
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        return model

    def __to_dataset(self, holdout_set):
        dataset = convert_samples_to_dataset(holdout_set, transform=transforms.ToTensor())
        return DataLoader(dataset, batch_size=25, shuffle=False)

    def __write_results(self, path, result_str):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        with open(path, 'w') as f:
            f.write(result_str)

        logger.info("Wrote results output to: {}".format(path))

