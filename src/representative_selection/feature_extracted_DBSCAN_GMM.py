import logging
from typing import List

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin

import torch
from torchvision import models, transforms

from ..datastructures import DataSample
from ..image_utils import load_rgb_image
from ..interfaces import RepresentativeSelection

logger = logging.getLogger(__name__)

class feature_extracted_DBSCAN_GMM(RepresentativeSelection):

    def __init__(self,
                 num_representatives: int=5,
                 DBSCAN_eps: float = 0.5,
                 DBSCAN_min_samples: int = 5,
                 class_whitelist: List[str]=None):
        """ First extract features from the image classes, then apply DBSCAN
            clustering. Based on the cardinality of each cluster, apply GMM
            with different n-component, select the mean.
            Finally, a pairwise Euclidean is used for finding images
            nearest to the cluster means.

        Parameters:
            num_representatives (int): Number of representatives to choose from each class
            class_whitelist (List[str]): Whitelist for representative selection. If none, it means no whitelist (default=None)
        """
        logger.debug('Creating feature_extracted_DBSCAN_GMM.')
        self.num_representatives = num_representatives
        self.class_whitelist = set(class_whitelist) if class_whitelist else None
        self.DBSCAN = DBSCAN(eps = DBSCAN_eps, min_samples = DBSCAN_min_samples)
        self.vgg19 = models.vgg19(pretrained=True)
        if torch.cuda.is_available():
            self.vgg19 = self.vgg19.cuda()

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

        logger.debug('---- Extracting features for images...')
        extracted_features = self.__extract_features(loaded)
        f_shape = extracted_features.shape
        # Flatten features to a vector
        features = np.reshape(extracted_features, (f_shape[0], f_shape[1]*f_shape[2]*f_shape[3]))

        logger.debug('---- Cluster phase 1...')
        # normalize the magnitude of features to [0,1], i dont know if the output will be negative...
        features = (features - np.min(features)) / (np.max(features) - np.min(features))

        self.DBSCAN.fit(features)
        #note: DBSCAN will generate different group label each time,
        #but the cluster is the same, thus the result is the same
        #So all these values are not saved in the class for further access
        cluster_phase_1_label = self.DBSCAN.labels_
        rep_distribution = self.__get_rep_distribution(N_represenative = self.num_representatives, label = cluster_phase_1_label)
        cluster_p1 = self.__get_sub_cluster_p1(data = extracted_features, label = cluster_phase_1_label)

        logger.debug('---- Cluster phase 2...')
        rep_list = []
        #for each cluster
        for i in range(len(cluster_p1)):

            sub_cluster = cluster_p1[i]
            num_rep = rep_distribution[i]

            #run a GMM to find the mean
            GMM = GaussianMixture(n_components = num_rep)
            GMM.fit(sub_cluster)
            means = GMM.means_

            #find the closest image
            reps = __find_closest(cluster = sub_cluster, means = means)
            #add them to the closest image
            rep_list += list(np.where(X_extracted == rep)[0][1] for rep in reps)

        rep_list = np.random.choice(np.array(rep_list), size = self.num_representatives, replace = False)

        return [paths[x] for x in rep_list]

    def __load_images(self, data):
        result = torch.stack([self.normalized(load_rgb_image(x)) for x in data])
        if torch.cuda.is_available():
            result = result.cuda()
        return result

    def __extract_features(self, data):
        features = []
        self.vgg19.eval()
        batch_size = 30
        for i in range(0, data.shape[0], batch_size):
            logger.debug('Extracting features from batch: {} (batch size = {}, total data = {})'.format(i / batch_size, batch_size, data.shape[0]))
            batch = data[i:(i+batch_size)]
            with torch.no_grad():
                features.append(self.vgg19.features(batch).data.cpu().detach().numpy())
        return np.vstack(features)


    def __get_rep_distribution(self, N_represenative, label):
        """
        given the cluster label generated by DBSCAN, calculate the number of representative for each cluster

        input:
            label: numpy.ndarray, the output of sklearn.cluster.DBSCAN's label_ method
        output:
            rep_distribution: numpy.ndarray, the number of representative for each cluster, in the same order
        """
        label_, count = np.unique(label, return_counts = True)
        #remove the noise term(cluster = -1)
        count = count[1:]
        rep_distribution = np.ceil(count*N_represenative/count.sum()).astype("int")
        return rep_distribution


    def __get_sub_cluster_p1(self, data, label):
        """
        given the original dataset and the label from sklearn.cluster, return the list of each cluster

        input:
            data: numpy.ndarray, original data passed into sklearn.cluster.DBSCAN's fit method
            label: numpy.ndarray, the output of sklearn.cluster.DBSCAN's label_ method
        output:
            cluster_p1: list of numpy.ndarray, the feature divided by labels
        """
        unique_label = np.unique(label)[1:]
        cluster_p1 = [[] for i in np.arange(unique_label.shape[0])]
        for i in np.arange(np.shape(label)[0]):
            if (label[i] != -1):
                cluster_p1[label[i]].append(data[i])
        cluster_p1 = [np.array(i) for i in cluster_p1]
        return cluster_p1

    def __find_closest(self, cluster, means):
        """
        find the single sample(image) in the cluster which has smallest euclidean distance to mean

        input:
            cluster: numpy.ndarray, a set of sample(image)
            mean: numpy.ndarray, an array of means, from the output of sklearn.mixture.GMM.means_

        output:
            rep: numpy.ndarray, the sample in the cluster which has smallest euclidean distance to each mean respectively
        """

        rep_index = pairwise_distances_argmin(means,cluster)
        return cluster[rep_index]
