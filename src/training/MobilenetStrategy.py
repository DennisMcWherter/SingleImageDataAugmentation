import logging
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from ..interfaces import TrainingStrategy
from ..torch_utils import convert_samples_to_dataset

logger = logging.getLogger(__name__)

class MobilenetV2Strategy(TrainingStrategy):

    def __init__(self, output_path, num_classes, num_epochs=100):
        self.output_path = output_path
        self.num_epochs = num_epochs
        self.num_classes = num_classes

        self.model = models.mobilenet_v2(pretrained=False, num_classes=num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
    
    def train(self, train_set, test_set):

        # NOTE: This assumes data can fit into memory. We can try a different approach if necessary
        train_inputs = self.__load_dataset(train_set)
        test_inputs = self.__load_dataset(test_set)

        for epoch in range(self.num_epochs):
            start = time.time()
            logger.info('Training epoch {}/{}'.format(epoch + 1, self.num_epochs))

            epoch_loss = 0.0
            for i, (inputs, labels) in enumerate(train_inputs):
                self.model.train()

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                if i % 2 == 0:
                    logger.info('---- Training Loss: {}'.format(epoch_loss / (i + 1.)))

            end = time.time()

            epoch_avg_loss = epoch_loss / len(train_inputs)
            test_loss = self.__test(test_inputs)

            logger.info('-- Epoch training loss: {}, test loss: {} (runtime: {} seconds)'.format(epoch_avg_loss, test_loss, (end - start)))

        logger.info('Saving trained model to: {}'.format(self.output_path))
        
        torch.save(self.model.state_dict(), self.output_path)

        return self.output_path

    def __test(self, test_inputs):
        for i, (inputs, labels) in enumerate(test_inputs):
            self.model.eval()

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)

            return loss.item()

    def __load_dataset(self, samples):
        dataset = convert_samples_to_dataset(samples, transform=transforms.ToTensor())
        return DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)

