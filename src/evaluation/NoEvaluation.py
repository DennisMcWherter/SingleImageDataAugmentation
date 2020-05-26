from ..interfaces import EvaluationStrategy

class NoEvaluation(EvaluationStrategy):

    def __init__(self, result_path):
        self.result_path = result_path

    def evaluate(self, holdout_set, model_path):
        return self.result_path
