import logging
import sys

from src.prebuilt_pipelines import FirstSinGANPipeline

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

FirstSinGANPipeline.restore_pipeline(representatives=False)
FirstSinGANPipeline.execute()

