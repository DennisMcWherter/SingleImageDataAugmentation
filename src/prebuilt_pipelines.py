from .pipeline import Pipeline

from .datasets import ImbalancedDataset, PassThroughDataset, TinyImageNet, CIFAR10
from .representative_selection import SelectAll, FeatureExtractedKMeansClusters
from .augmentations import NoAugmentation, PrecomputedAugmentation, SinGANAugmentation
from .training import NoTraining, MobilenetStrategy
from .evaluation import NoEvaluation, MobilenetEvaluation

# Tiny imagenet
tinyImageNet = TinyImageNet.TinyImageNetDataset('tiny-imagenet-200', num_classes=30)
imbalancedTinyImageNet = ImbalancedDataset.ImbalancedDataset(tinyImageNet)

# CIFAR-10
cifar10 = CIFAR10.CIFAR10()
imbalancedCIFAR10 = ImbalancedDataset.StratifiedRandomImbalancedDataset(cifar10, num_labels=3)

# Test pipeline that does nothing
TestPipeline = Pipeline(pipeline_name="test_pipeline",
                        dataset=PassThroughDataset.PassThroughDataset(),
                        selection_strategy=SelectAll.SelectAllSelection(),
                        augmentation_strategy=NoAugmentation.NoAugmentation(),
                        training_strategy=NoTraining.NoTraining('test_model'),
                        evaluation_strategy=NoEvaluation.NoEvaluation('result_path'))

# Imbalanced pipeline that performs no augmentation on CIFAR-10
ImbalancedNoAugCIFAR10 = Pipeline(pipeline_name="imbalanced_no_aug_cifar10",
                                  dataset=imbalancedCIFAR10,
                                  selection_strategy=SelectAll.SelectAllSelection(),
                                  augmentation_strategy=NoAugmentation.NoAugmentation(),
                                  training_strategy=MobilenetStrategy.MobilenetV2Strategy('output/mobilenet_imbalanced_no_aug_cifar10/model', num_classes=10),
                                  evaluation_strategy=MobilenetEvaluation.MobilenetV2EvaluationStrategy(output_path='output/mobilenet_imbalanced_no_aug_cifar10/results', num_classes=10))

# Imbalanced pipeline that performs no augmentation
ImbalancedNoAug = Pipeline(pipeline_name="imbalanced_no_aug",
                           dataset=imbalancedTinyImageNet,
                           selection_strategy=SelectAll.SelectAllSelection(),
                           augmentation_strategy=NoAugmentation.NoAugmentation(),
                           training_strategy=MobilenetStrategy.MobilenetV2Strategy('output/mobilenet_imbalanced_no_aug/model', num_classes=30),
                           evaluation_strategy=MobilenetEvaluation.MobilenetV2EvaluationStrategy(output_path='output/mobilenet_imbalanced_no_aug/results', num_classes=30))

# First pipeline
FirstSinGANPipeline = Pipeline(pipeline_name="first_singan_pipeline",
                               dataset=imbalancedTinyImageNet,
                               selection_strategy=FeatureExtractedKMeansClusters.FeatureExtractedKMeansClusters(num_representatives=2, class_whitelist=[1,2]),
                               augmentation_strategy=SinGANAugmentation.SinGANAugmentation('./SinGANSource', 'first_singan_pipeline/augmentation'),
                               training_strategy=MobilenetStrategy.MobilenetV2Strategy('output/mobilenet_first_singan/model', num_classes=30),
                               evaluation_strategy=MobilenetEvaluation.MobilenetV2EvaluationStrategy(output_path='output/mobilenet_first_singan/results', num_classes=30))

