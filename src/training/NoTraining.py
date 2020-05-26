from ..interfaces import TrainingStrategy

class NoTraining(TrainingStrategy):

    def __init__(self, model_path):
        self.model_path = model_path

    def train(self, train_set, test_set):
        return self.model_path
