import logging
import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from ..interfaces import TrainingStrategy
from ..models.MobilenetV2 import TestMobilenetV2
from ..torch_utils import compute_accuracy, convert_samples_to_dataset, test_model

logger = logging.getLogger(__name__)

class MobilenetV2Strategy(TrainingStrategy):

    def __init__(self, output_path, num_classes, num_epochs=50):
        self.output_path = output_path
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.model = TestMobilenetV2(num_classes)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
    
    def train(self, train_set, test_set):

        # NOTE: This assumes data can fit into memory. We can try a different approach if necessary
        train_inputs = self.__load_dataset(train_set)
        test_inputs = self.__load_dataset(test_set)

        for epoch in range(self.num_epochs):
            start = time.time()
            logger.info('Training epoch {}/{}'.format(epoch + 1, self.num_epochs))

            epoch_total_accuracy = 0.0
            epoch_total_samples = 0.0
            epoch_loss = 0.0
            for i, (inputs, labels) in enumerate(train_inputs):
                epoch_total_samples += len(labels)

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                self.model.train()

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                batch_accuracy = compute_accuracy(outputs, labels)
                epoch_total_accuracy += batch_accuracy

                epoch_loss += loss.item()
                if i % 20 == 0:
                    logger.info('---- Training Loss: {}, Training Accuracy: {}'.format(epoch_loss / (i + 1.), batch_accuracy))

            end = time.time()

            epoch_avg_loss = epoch_loss / len(train_inputs)
            epoch_avg_accuracy = epoch_total_accuracy / epoch_total_samples
            test_loss, test_accuracy = test_model(self.model, self.loss_fn, test_inputs)

            logger.info('-- Epoch {} Results (runtime: {} seconds) --'.format(epoch + 1, (end - start)))
            logger.info('---- Epoch training loss: {}, training accuracy: {}'.format(epoch_avg_loss, epoch_avg_accuracy))
            logger.info('---- Epoch test loss: {}, test accuracy: {}'.format(test_loss, test_accuracy))

        logger.info('Saving trained model to: {}'.format(self.output_path))

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        
        model_path = os.path.join(self.output_path, 'test-mobilenet.pt')
        torch.save(self.model.state_dict(), model_path)

        return model_path

    def __load_dataset(self, samples):
        dataset = convert_samples_to_dataset(samples, transform=transforms.ToTensor())
        return DataLoader(dataset, batch_size=25, shuffle=False, num_workers=4)

