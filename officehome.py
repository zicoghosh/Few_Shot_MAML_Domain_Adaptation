# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 20:40:57 2021

@author: ghosh
"""

"""
Loading and using the Home-Office dataset.

To use these APIs, you should prepare a directory that
contains three sub-directories: train, test, and val.
Each of these three directories should contain one
sub-directory per WordNet ID.
"""

import os
import random

from PIL import Image
import numpy as np
import torch

from augment_data import rand_augment_transform

def read_dataset(data_dir):
    """
    Read the Home-Office dataset.

    Args:
      data_dir: directory containing Mini-ImageNet.

    Returns:
      A tuple (train, val, test) of sequences of
        OfficeHomeClass instances. 
    """
    return tuple(_read_classes(os.path.join(data_dir, x)) for x in ['train', 'val', 'test'])

def _read_classes(dir_path):
    """
    Read the WNID directories in a directory.
    """
    return [OfficeHomeClass(os.path.join(dir_path, f)) for f in os.listdir(dir_path)]

def augment_dataset(dataset):
    """
    Augment the dataset by adding 90 degree rotations.

    Args:
      dataset: an iterable of Characters.

    Returns:
      An iterable of augmented Characters.
    """
    for image_obj in dataset:
        yield OfficeHomeClass(image_obj.dir_path, rotation=0, rand = True)

# pylint: disable=R0903
class OfficeHomeClass:
    """
    A single image class.
    """
    def __init__(self, dir_path, rotation=0, rand = False):
        self.dir_path = dir_path
        self.rotation = rotation
        self.rand_ = rand
        _FILL = (128, 128, 128)
        self._HPARAMS_DEFAULT = dict(
            translate_const=250,
            img_mean=_FILL,
        )
        self.tf = rand_augment_transform('rand-m9-n3-mstd0.55', self._HPARAMS_DEFAULT)
        self._cache = {}

    def sample(self, num_images, domain):
        """
        Sample images (as pytorch tensor) from the class.

        Returns:
          A sequence of 3 * 84 * 84 tensors.
          Each pixel ranges from 0 to 1.
        """
        directory = os.path.join(self.dir_path, domain)
        names = [f for f in os.listdir(directory) if f.endswith('.jpg')]
        random.shuffle(names)
        images = []
        for name in names[:num_images]:
            images.append(self._read_image(name, directory))
        return images

    def _read_image(self, name, directory):
        if name in self._cache:
            return self._cache[name]
        with open(os.path.join(directory, name), 'rb') as in_file:
            img = Image.open(in_file).resize((84, 84)).convert('RGB').rotate(self.rotation)
            if(self.rand_):
                img = self.tf(img)    
            img = np.array(img).astype('float32') / 0xff
            img =np.rollaxis(img, 2, 0)
            self._cache[name] = torch.tensor(img)
            return self._read_image(name, directory)
    
    def _show_image(self):
        im = Image.open(self.dir_path)
        im.show()
        return