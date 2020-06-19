import cv2
import logging
import numpy as np
import os
from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from ..datastructures import DataSample
from ..interfaces import Dataset

logger = logging.getLogger(__name__)

class CIFAR10(Dataset):

    def __init__(self, path='./data_generating/cifar-10-batches-py', expanded_path='./cifar10', test_pct=0.1):
        """ CIFAR-10 dataset.

        Parameters:
            path (str): Path to CIFAR-10 dataset (pickled from original source)
            expanded_path (str): Path for the expanded CIFAR 10. If directory exists, data will not be expanded further.
            test_pct (float): Test data percentage (default=0.1)
        """
        self.path = path
        self.expanded_path = expanded_path
        self.test_pct = test_pct

    def prepare(self):
        logger.debug('-- Loading CIFAR-10 dataset...')

        if not os.path.exists(self.expanded_path):
            logger.debug('--- CIFAR-10 output path does not exist. Expanding to: {} from original path: {}'.format(self.expanded_path, self.path))

            X_train, y_train = self.__load_cifar10()
            X_holdout, y_holdout = self.__load_cifar10(glob='test_batch*')

            train_path = os.path.join(self.expanded_path, 'train')
            self.__write_images(train_path, X_train, y_train[0], y_train[1])

            test_path = os.path.join(self.expanded_path, 'test')
            self.__write_images(test_path, X_holdout, y_holdout[0], y_holdout[1])

        logger.debug('--- Loading expanded CIFAR-10 from: {}'.format(self.expanded_path))
        self.__load_expanded()

        # One-hot encode labels
        all_labels = [x.get_label() for x in self.train_data]
        encoder = OneHotEncoder(sparse=False)
        sorted_labels = sorted(list(set(all_labels)))
        encoder_labels = [[sorted_labels[i]] for i in range(len(sorted_labels))]
        encoder.fit(encoder_labels)
        transformed_labels = encoder.transform(encoder_labels).argmax(axis=1)

        all_labels = encoder.transform([[x] for x in all_labels]).argmax(axis=1)

        self.encoding = dict((sorted_labels[i], int(transformed_labels[i])) for i in range(len(sorted_labels)))

        logger.debug('--- Done loading CIFAR-10')

    def get_training_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_holdout_data(self):
        return self.holdout_data

    def get_encoding(self):
        return self.encoding

    def __load_cifar10(self, glob='data_batch_*'):
        X_loaded = []
        y_loaded = []
        filenames = []
        for batch in Path(self.path).glob(glob):
            with open(batch, 'rb') as f:
                dict = pickle.load(f, encoding='bytes')
                data = dict[b'data']
                labels = dict[b'labels']
                names = dict[b'filenames']
                X_loaded.append(data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1))
                y_loaded.append(labels)
                filenames.append(names)
        X = np.concatenate(X_loaded)
        y = np.concatenate(y_loaded)
        names = np.concatenate(filenames)
        result_labels = np.vstack((y, names))
        return (X, result_labels)

    def __write_images(self, path, X, y, filenames):
        if not os.path.exists(path):
            os.makedirs(path)

        assert(X.shape[0] == y.shape[0] == filenames.shape[0])

        logger.debug('--- Writing expanded CIFAR-10 images to disk.')
        for x in range(len(filenames)):
            labeled_dir = os.path.join(path, y[x].decode('utf-8'))
            if not os.path.exists(labeled_dir):
                os.makedirs(labeled_dir)
            filepath = os.path.join(labeled_dir, filenames[x].decode('utf-8'))
            bgr = cv2.cvtColor(X[x], cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, bgr)

    def __load_expanded(self):
        train_path = os.path.join(self.expanded_path, 'train')
        test_path = os.path.join(self.expanded_path, 'test')

        training_data = self.__load_samples(train_path)
        np.random.shuffle(training_data)
        
        num_test_samples = int(self.test_pct * len(training_data))

        self.train_data = training_data[num_test_samples:]
        self.test_data = training_data[:num_test_samples]
        self.holdout_data = self.__load_samples(test_path)

    def __load_samples(self, path):
        samples = []
        dirs = os.listdir(path)
        for label in dirs:
            for img_path in Path(os.path.join(path, label)).glob('*.png'):
                samples.append(DataSample(path=str(img_path), label=label))
        return samples
