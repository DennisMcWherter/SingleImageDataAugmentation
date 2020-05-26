from ..interfaces import RepresentativeSelection

class SelectAllSelection(RepresentativeSelection):

    def select_samples(self, input_dataset):
        return input_dataset
