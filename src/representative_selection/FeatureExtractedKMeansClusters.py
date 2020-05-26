import logging
from typing import List

import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

import torch
from torchvision import models, transforms

from ..datastructures import DataSample
from ..image_utils import load_rgb_image
from ..interfaces import RepresentativeSelection

logger = logging.getLogger(__name__)

class FeatureExtractedKMeansClusters(RepresentativeSelection):

    def __init__(self, num_representatives: int=5, class_whitelist: List[str]=None):
        """ First extract features from the image classes, then apply K-means
            clustering. Finally, a pairwise Euclidean is used for finding images
            nearest to the cluster centers.

        Parameters:
            num_representatives (int): Number of representatives to choose from each class
            class_whitelist (List[str]): Whitelist for representative selection. If none, it means no whitelist (default=None)
        """
        logger.debug('Creating FeatureExtractKMeansClusters.')
        self.num_representatives = num_representatives
        self.class_whitelist = set(class_whitelist) if class_whitelist else None
        self.vgg19 = models.vgg19(pretrained=True)

        # ToTensor will convert an RGB image (HxWxC) with values [0, 255] to
        # an RGB tensor of (CxHxW) with values [0,1]
        toTensor = transforms.ToTensor()

        # Normalization values come from docs:
        # https://pytorch.org/docs/stable/torchvision/models.html
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.normalized = transforms.Compose([toTensor, normalize])

    def select_samples(self, input_dataset):
        filtered = [x for x in input_dataset if not self.class_whitelist or x.get_label() in self.class_whitelist]
        logger.debug('-- Filtered to class whitelist: {}. {} samples remaining.'.format(self.class_whitelist, len(filtered)))

        partitioned = self.__partition_by_class(filtered)

        representatives = []

        for img_class in partitioned.keys():
            paths = partitioned[img_class]
            logger.debug('---- Loading {} images for class: {}'.format(len(paths), img_class))

            representatives.extend([DataSample(x, img_class) for x in self.__find_representatives(paths)])

            logger.debug('---- Extracting features for images...')

        return representatives

    def __partition_by_class(self, data):
        partitioned = {}
        for sample in data:
            key = sample.get_label()
            value = sample.get_path()
            if key not in partitioned:
                partitioned[key] = []
            partitioned[key].append(value)
        return partitioned

    def __find_representatives(self, paths):
        loaded = self.__load_images(paths)
        features = self.__extract_features(loaded)

        kmeans = KMeans(n_clusters=self.num_representatives, random_state=0).fit(features)
        
        representatives = pairwise_distances_argmin(kmeans.cluster_centers_, features)

        return [paths[x] for x in representatives]

    def __load_images(self, data):
        return torch.stack([self.normalized(load_rgb_image(x)) for x in data])

    def __extract_features(self, data):
        features = []
        self.vgg19.eval()
        batch_size = 30
        for i in range(0, data.shape[0], batch_size):
            logger.debug('Extracting features from batch: {} (batch size = {}, total data = {})'.format(i / batch_size, batch_size, data.shape[0]))
            batch = data[i:(i+batch_size)]
            features.append(self.vgg19(batch).data.detach().numpy())
        return np.vstack(features)

