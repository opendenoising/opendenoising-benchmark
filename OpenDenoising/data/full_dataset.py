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


class FullDatasetGenerator(AbstractDatasetGenerator):
    """Dataset generator based on Keras library. This class is used for non-blind denoising problems. Unlike
    ClenDatasetGenerator class, this class corresponds to the case where both clean and noisy samples are available and
    paired (for each noisy image, there is one and only one clean image with same filename).

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
    """
    def __init__(self, path, batch_size=32, shuffle=True, name="FullDataset", n_channels=1, preprocessing=None):
        super().__init__(path, batch_size, shuffle, name, n_channels)
        self.n_channels = n_channels
        self.filenames = np.array(os.listdir(os.path.join(self.path, "in")))
        self.preprocessing = [] if preprocessing is None else preprocessing
        self.on_epoch_end()
        module_logger.info("Generating data from {}".format(os.path.join(self.path)))

    def __getitem__(self, i):
        """ Generate batches of data """
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
            List of strings containing filenames to read. Note that, for each noisy image filename there must be a clean
            image with same filename.

        Returns
        -------
        noisy_batch : :class:`numpy.ndarray`
            Batch of noisy images.
        clean_batch : :class:`numpy.ndarray`
            Batch of reference images.
        """
        # Noised image and ground truth initialization
        ref_batch = []
        inp_batch = []

        for filename in batch_filenames:
            # Compose path
            clean_filepath = os.path.join(self.path, "in", filename)
            noisy_filepath = os.path.join(self.path, "ref", filename)
            # Read images
            ref = imread(clean_filepath)
            ref = img_as_float32(ref)
            inp = imread(noisy_filepath)
            inp = img_as_float32(inp)

            if not inp.shape == ref.shape:
                raise ValueError("Expected {} to have same shape of {}, but got {} and {}".format(clean_filepath,
                                                                                                  noisy_filepath,
                                                                                                  ref.shape, inp.shape))
            # Corrects shape of reference
            if ref.ndim == 3 and ref.shape[-1] == 3 and self.n_channels == 1:
                # Converts RGB to Gray
                ref = rgb2gray(ref)
            if ref.ndim == 2 and self.n_channels == 1:
                # Expand last dim if image is grayscale
                ref = np.expand_dims(ref, axis=-1)
            elif ref.ndim == 2 and self.n_channels == 3:
                raise ValueError("Expected RGB image but got Grayscale (image shape: {})".format(ref.shape))

            # Corrects shape of input image
            if inp.ndim == 3 and inp.shape[-1] == 3 and self.n_channels == 1:
                # Converts RGB to Gray
                inp = rgb2gray(inp)
            if inp.ndim == 2 and self.n_channels == 1:
                # Expand last dim if image is grayscale
                inp = np.expand_dims(inp, axis=-1)
            elif inp.ndim == 2 and self.n_channels == 3:
                raise ValueError("Expected RGB image but got Grayscale (image shape: {})".format(inp.shape))

            # Apply preprocessing functions
            for func in self.preprocessing:
                inp, ref = func(inp, ref)

            # Append images to lists
            ref_batch.append(ref)
            inp_batch.append(inp)
        ref_batch = np.array(ref_batch) if ref_batch[0].ndim == 3 else np.concatenate(ref_batch, axis=0)
        inp_batch = np.array(inp_batch) if inp_batch[0].ndim == 3 else np.concatenate(inp_batch, axis=0)
        module_logger.debug("Data shape: {}".format(ref_batch.shape))

        return ref_batch, inp_batch

    def __next__(self):
        """Returns image batches sequentially."""
        while True:
            for input_batch, output_batch in self:
                return input_batch, output_batch

    def __repr__(self):
        return "Dataset name: {}, Dataset type: {}, Path: {}, " \
               "Batch Size: {}, preprocessing: {}".format(self, "Full", self.path, self.batch_size,
                                                          self.preprocessing)
