import logging
import sys

import numpy as np

import torch

from src.prebuilt_pipelines import *

# Set for reproducibility
np.random.seed(123)
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Application log settings
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

#FirstSinGANPipeline.restore_pipeline(representatives=True, training=True, augmentation=False)
#FirstSinGANPipeline.execute()
ImbalancedNoAug.execute()

