import logging
from typing import List

import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin

from ..datastructures import DataSample
from ..image_utils import load_rgb_image
from ..interfaces import RepresentativeSelection

logger = logging.getLogger(__name__)

class PCA_DBSCAN_GMM(RepresentativeSelection):

    def __init__(self,
                 num_represenatives: int = 40,
                 PCA_N_components: int = 5,
                 DBSCAN_eps: float = 0.5,
                 DBSCAN_min_samples: int = 5,
                 class_whitelist: List[str] = None):
        """
        The representative selection procedure
        it run 1. PCA to original data
               2. DBSCAN for cluster phase 1
               3. Gaussian mixture model on each cluster in phase 1
               4. Find the point with smallest euclidean distance to these means
        and returns the index of these points in original data

        input:
            data: numpy.ndarray, the dataset of image, in current setting, each image need to be flattened (sklearn.PCA only takes flattened value)
            N_represenative: int, the minimal value of representative, the final result may have higher number
            PCA_N_components: int, the number of component for PCA
            DBSCAN_eps: float, the eps hyperparameter fpr DBSCAN
            DBSCAN_min_samples: int

        output:
            rep_list: list of int, the index of representatives in the dataset
                      (which can be use as index of np.array directly)
        """
        logger.debug('Creating PCA_DBSCAN_GMM_Clusters.')
        self.num_representatives = num_representatives
        self.PCA = PCA(n_components=PCA_N_components)
        self.DBSCAN = DBSCAN(eps = DBSCAN_eps, min_samples = DBSCAN_min_samples)
        self.class_whitelist = set(class_whitelist) if class_whitelist else None

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

        logger.debug('---- Cluster phase 1...')
        self.DBSCAN.fit(extracted_features)
        #note: DBSCAN will generate different group label each time,
        #but the cluster is the same, thus the result is the same
        #So all these values are not saved in the class for further access
        cluster_phase_1_label = self.DBSCAN.labels_
        rep_distribution = self.get_rep_distribution(N_represenative = self.num_representatives, label = cluster_phase_1_label)
        cluster_p1 = self.get_sub_cluster_p1(data = extracted_features, label = cluster_phase_1_label)

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
            reps = find_closest(cluster = sub_cluster, means = means)
            #add them to the closest image
            rep_list += list(np.where(X_extracted == rep)[0][1] for rep in reps)

        rep_list = np.random.choice(np.array(rep_list), size = self.num_representatives, replace = False)

        return [paths[x] for x in rep_list]

    def __load_images(self, data):
        result = np.vstack([load_rgb_image(x).flatten() for x in data])
        return result

    def __extract_features(self, data):
        """
        given the Data, run PCA

        input:
            data: numpy.ndarray, original [0,255] flattened data

        output:
            features: numpy.ndarray, the PCA extracted features,
                      dimensions are specified in the intialization
        """
        data /= np.max(data)
        features = self.PCA.fit_transform(data)
        return features

    def get_rep_distribution(self, N_represenative, label):
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


    def get_sub_cluster_p1(self, data, label):
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

    def find_closest(self, cluster, means):
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
