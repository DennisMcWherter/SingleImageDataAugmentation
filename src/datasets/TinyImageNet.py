import logging
import numpy as np
import os
from pathlib import Path

from sklearn.model_selection import train_test_split

from ..datastructures import DataSample
from ..interfaces import Dataset

class TinyImageNetDataset(Dataset):

    def __init__(self, path, num_classes=200, holdout_pct=0.1, test_pct=0.1):
        self.num_classes = num_classes
        self.path = path
        self.holdout_pct = holdout_pct
        self.test_pct = test_pct

    def prepare(self):
        logging.debug('-- Loading TinyImageNet with {} classes'.format(self.num_classes))
        
        paths = [str(x) for x in Path(self.path).glob(os.path.join('train', '**', '*.JPEG'))]
        data = [DataSample(path=x, label=self.__path_to_label(x)) for x in paths]
        
        selected_classes = list(set([x.get_label() for x in data]))
        np.random.shuffle(selected_classes)
        selected_classes = set(selected_classes[:self.num_classes])

        filtered_data = [x for x in data if x.get_label() in selected_classes]
        (all_paths, all_labels) = ([x.get_path() for x in filtered_data], [x.get_label() for x in filtered_data])
        X_all_train, X_holdout, y_all_train, y_holdout = train_test_split(all_paths, all_labels, test_size=self.holdout_pct)
        X_train, X_test, y_train, y_test = train_test_split(X_all_train, y_all_train, test_size=self.test_pct)

        self.holdout_data = [DataSample(path=X_holdout[i], label=y_holdout[i]) for i in range(len(X_holdout))]
        self.train_data = [DataSample(path=X_train[i], label=y_train[i]) for i in range(len(X_train))]
        self.test_data = [DataSample(path=X_test[i], label=y_test[i]) for i in range(len(y_test))]

        logging.debug('-- Loaded {} samples'.format(len(self.holdout_data) + len(self.train_data) + len(self.test_data)))

    def get_training_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_holdout_data(self):
        return self.holdout_data

    def __path_to_label(self, path):
        return path.split(os.path.sep)[-1].split('_')[0]


