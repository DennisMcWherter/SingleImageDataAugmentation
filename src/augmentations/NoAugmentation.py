from ..interfaces import AugmentationStrategy

class NoAugmentation(AugmentationStrategy):

    def augment_data(self, dataset, representatives):
        return []
