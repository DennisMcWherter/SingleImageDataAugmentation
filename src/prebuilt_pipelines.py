from .pipeline import Pipeline

from .datasets import PassThroughDataset, TinyImageNet
from .representative_selection import SelectAll, FeatureExtractedKMeansClusters
from .augmentations import NoAugmentation, PrecomputedAugmentation
from .training import NoTraining, MobilenetStrategy
from .evaluation import NoEvaluation, MobilenetEvaluation

# Test pipeline that does nothing
TestPipeline = Pipeline(pipeline_name="test_pipeline",
                        dataset=PassThroughDataset.PassThroughDataset(),
                        selection_strategy=SelectAll.SelectAllSelection(),
                        augmentation_strategy=NoAugmentation.NoAugmentation(),
                        training_strategy=NoTraining.NoTraining('test_model'),
                        evaluation_strategy=NoEvaluation.NoEvaluation('result_path'))

# First pipeline
FirstSinGANPipeline = Pipeline(pipeline_name="first_singan_pipeline",
                               dataset=TinyImageNet.TinyImageNetDataset('tiny-imagenet-200', num_classes=30),
                               selection_strategy=FeatureExtractedKMeansClusters.FeatureExtractedKMeansClusters(),
                               augmentation_strategy=PrecomputedAugmentation.PrecomputedAugmentation('./intermediate/first_singan_pipeline/test_data.csv'),
                               training_strategy=MobilenetStrategy.MobilenetV2Strategy('output/mobilenet_first_singan/model', num_classes=30),
                               evaluation_strategy=MobilenetEvaluation.MobilenetV2EvaluationStrategy(num_classes=30))

