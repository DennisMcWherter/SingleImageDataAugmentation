import logging
import os
import subprocess
import sys

from shutil import copy, rmtree

from pathlib import Path

import torch

from ..datastructures import DataSample
from ..interfaces import AugmentationStrategy

logger = logging.getLogger(__name__)


class ConSinGANAugmentation(AugmentationStrategy):

    def __init__(self, consingan_root, output_path):
        self.output_path = output_path
        self.consingan_root = consingan_root

    def augment_data(self, dataset, reps):
        # Free any cached GPU memory for SinGAN
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # TODO: Do for all representatives when testing is finished
        return self.__generate_from_representatives(reps)

    def __generate_from_representatives(self, reprs):
        # clean any old data first
        if os.path.exists(self.output_path):
            rmtree(self.output_path)
        os.makedirs(self.output_path)

        consingan_raw_output_path = os.path.join(self.output_path, 'consingan_raw_output')
        if not os.path.exists(consingan_raw_output_path):
            os.makedirs(consingan_raw_output_path)

        script_path = os.path.join(self.consingan_root, 'main_train.py')

        results = []
        for sample in reprs:
            labeled_output_dir = os.path.join(self.output_path,
                                              str(sample.get_label())) + os.path.sep
            if not os.path.exists(labeled_output_dir):
                os.makedirs(labeled_output_dir)

            args = [
                sys.executable,
                script_path,
                "--gpu 0",
                "--train_mode generation",
                "--input_name",
                os.path.join(os.getcwd(), sample.get_path())
                #  "--lr_scale 0.1",
                #  "--train_stages 6"
            ]
            print(' '.join(args))
            run = subprocess.run(args=args,
                                 cwd=os.getcwd())

            if not run.returncode == 0:
                logger.error("Failed to generate ConSinGAN sample for: {}".format(sample.get_path()))
                logger.error(run.stdout)
                raise BaseException("Failed to generate ConSinGAN sample")

            # TODO: We might be able to get rid of most of this if the --out switch works how we think it does.
            consingan_generated_output = os.path.join(consingan_raw_output_path, 'RandomSamples',
                                                      os.path.splitext(sample.get_path())[0] + '.')

            for generated in Path(consingan_generated_output).glob(os.path.join('**', '*.png')):
                generated_path = str(generated)
                generated_name = generated_path.split(os.path.sep)[-1]

                logger.debug('---- Copying generated output: {} to {}'.format(generated_path,
                                                                              labeled_output_dir))

                copy(generated_path, labeled_output_dir)

                results.append(DataSample(path=os.path.join(labeled_output_dir, generated_name),
                                          label=sample.get_label()))

            return results

