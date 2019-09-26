# Copyright or © or Copr. IETR/INSA Rennes (2019)
# 
# Contributors :
#     Eduardo Fernandes-Montesuma eduardo.fernandes-montesuma@insa-rennes.fr (2019)
#     Florian Lemarchand florian.lemarchand@insa-rennes.fr (2019)
# 
# 
# OpenDenoising is a computer program whose purpose is to benchmark image
# restoration algorithms.
# 
# This software is governed by the CeCILL-C license under French law and
# abiding by the rules of distribution of free software. You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL-C
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
# 
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author, the holder of the
# economic rights, and the successive licensors have only  limited
# liability.
# 
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
# 
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-C license and that you accept its terms.


import os
import numpy as np

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_float32
from OpenDenoising.data import module_logger
from OpenDenoising.data import AbstractDatasetGenerator


class BlindDatasetGenerator(AbstractDatasetGenerator):
    """Dataset generator based on Keras library. This class is used for Blind denoising problems, where only noisy
    images are available.

    Attributes
    ----------
    path : str
        String containing the path to image files directory.
    batch_size : int
        Size of image batch.
    n_channels : int
        1 for grayscale, 3 for RGB.
    shuffle : bool
        Whether to shuffle the dataset at each epoch or not.
    name : str
        String containing the dataset's name.
    preprocessing : list
        List of preprocessing functions, which will be applied to each image.
    target_fcn : function
        Function implementing how to generate target images from noisy ones.
    """
    def __init__(self, path, batch_size=32, shuffle=True, name="CleanDataset", n_channels=1, preprocessing=None,
                 target_fcn=None):
        super().__init__(path, batch_size, shuffle, name, n_channels)
        self.filenames = np.array(os.listdir(os.path.join(self.path)))
        self.preprocessing = [] if preprocessing is None else preprocessing
        self.on_epoch_end()
        self.target_fcn = target_fcn
        module_logger.info("Generating data from {}".format(os.path.join(self.path)))

    def __getitem__(self, i):
        """Generates image batches from filenames.

        Parameters
        ----------
        i : int
            Batch index to get.

        Returns
        -------
        inp : :class:`numpy.ndararray`
            Batch of noisy images.
        ref : :class:`numpy.ndarray`
            Batch of target images.
        """
        # Get batch_filenames
        batch_filenames = self.filenames[i * self.batch_size: (i + 1) * self.batch_size]
        module_logger.debug("[{}] Got following batch names: {}".format(self, batch_filenames))
        # Get data batches
        inp, ref = self.__data_generation(batch_filenames)
        return inp, ref

    def __data_generation(self, batch_filenames):
        """Data generation method

        Parameters
        ----------
        batch_filenames : list
            List of strings containing filenames to read.

        Returns
        -------
        noisy_batch : :class:`numpy.ndarray`
            Batch of noisy images.
        """
        # Noised image and ground truth initialization
        inp_batch = []
        ref_batch = []

        for filename in batch_filenames:
            filepath = os.path.join(self.path, filename)
            module_logger.debug("Loading image located on {}".format(filepath))
            inp = imread(filepath)
            inp = img_as_float32(inp)

            if inp.ndim == 3 and inp.shape[-1] == 3 and self.n_channels == 1:
                # Converts RGB to Gray
                inp = rgb2gray(inp)
            if inp.ndim == 2 and self.n_channels == 1:
                # Expand last dim if image is grayscale
                inp = np.expand_dims(inp, axis=-1)
            elif inp.ndim == 2 and self.n_channels == 3:
                raise ValueError("Expected RGB image but got Grayscale (image shape: {})".format(inp.shape))

            for func in self.preprocessing:
                # Preprocessing pipeline
                inp = func(inp)

            # Generates target from input
            inp, ref = self.target_fcn(inp)
            ref_batch.append(ref)
            inp_batch.append(inp)
        inp_batch = np.array(inp_batch)
        ref_batch = np.array(ref_batch)
        module_logger.debug("Data shape: {}".format(inp_batch.shape))
        return inp_batch, ref_batch

    def __repr__(self):
        return "Dataset name: {}, Dataset type: {}, Path: {}, " \
               "Batch Size: {}, preprocessing: {}".format(self, "Clean", self.path, self.batch_size,
                                                          self.preprocessing)
