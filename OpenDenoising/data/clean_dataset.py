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
from OpenDenoising.data.utils import corrupt
from OpenDenoising.data.utils import gaussian_noise
from OpenDenoising.data import AbstractDatasetGenerator


class CleanDatasetGenerator(AbstractDatasetGenerator):
    """Dataset generator based on Keras library. This class is used for non-blind denoising problems where only clean
    images are available. To use such dataset to train denoising networks, you need to specify a type of artificial
    noise that will be added to each clean image.

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
    channels_first : bool
        Whether data is formatted as (BatchSize, Height, Width, Channels) or (BatchSize, Channels, Height, Width).
    name : str
        String containing the dataset's name.
    preprocessing : list
        List of preprocessing functions, which will be applied to each image.
    noise_config : dict
        Dictionary whose keys are functions implementing the noise process, and the value is a list containing the noise
        function parameters. If you do not want to specify any parameters, your list should be empty.

    Examples
    --------
    
    The following example corresponds to a Dataset Generator which reads images from "./images", yields batches of
    length 32, applies Gaussian noise with intensity drawn uniformely from the range [0, 55] followed by
    "Super Resolution noise" of intensity 4. Moreover, the dataset shuffles the data, yields them in NHWC format,
    and does not apply any preprocessing function. **NOTE**: your list should be in the same order as your arguments.

    >>> from OpenDenoising import data
    >>> noise_config = {data.utils.gaussian_blind_noise: [0, 55],
    ...                 data.utils.super_resolution_noise: [4]}
    >>> datagen = data.CleanDatasetGenerator("./images", 32, noise_config, True, False, "MyData", 1, None)
    """
    def __init__(self, path, batch_size=32, noise_config=None, shuffle=True, name="CleanDataset", n_channels=1,
                 preprocessing=None):
        super().__init__(path, batch_size, shuffle, name, n_channels)
        if noise_config is None:
            noise_config = {
                gaussian_noise: [25]
            }
        self.noise_functions = noise_config.keys()
        self.noise_args = [noise_config[noise_type] for noise_type in noise_config]
        self.preprocessing = [] if preprocessing is None else preprocessing
        self.n_channels = n_channels
        self.filenames = np.array(os.listdir(os.path.join(self.path, "ref")))
        self.on_epoch_end()
        module_logger.info("Generating data from {}".format(os.path.join(self.path, 'ref',)))
        self.image_shape = self[0][0].shape
        module_logger.debug("[{}] Image shape: {}".format(self.name, self.image_shape))

    def __getitem__(self, i):
        """Generates batches of data."""
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
        inp_batch = []
        ref_batch = []

        for filename in batch_filenames:
            filepath = os.path.join(self.path, 'ref', filename)
            ref = imread(filepath)
            ref = img_as_float32(ref)

            if ref.ndim == 3 and ref.shape[-1] == 3 and self.n_channels == 1:
                # Converts RGB to Gray
                ref = rgb2gray(ref)
            if ref.ndim == 2 and self.n_channels == 1:
                # Expand last dim if image is grayscale
                ref = np.expand_dims(ref, axis=-1)
            elif ref.ndim == 2 and self.n_channels == 3:
                raise ValueError("Expected RGB image but got Grayscale (image shape: {})".format(ref.shape))
            inp = ref.copy()

            for noise_function, noise_arg in zip(self.noise_functions, self.noise_args):
                # Adds noise to the reference.
                inp = noise_function(inp, *noise_arg)

            # Applies preprocessing functions in order
            for func in self.preprocessing:
                inp, ref = func(inp, ref)

            ref_batch.append(ref)
            inp_batch.append(inp)
        inp_batch = np.stack(inp_batch)
        ref_batch = np.stack(ref_batch)
        if len(inp_batch.shape) > 4:
            inp_batch = inp_batch.reshape([-1, *inp_batch.shape[2:]])
            ref_batch = ref_batch.reshape([-1, *ref_batch.shape[2:]])
        return inp_batch, ref_batch

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Dataset name: {}, Dataset type: {}, Path: {}, " \
               "Batch Size: {}, preprocessing: {}, shape: {}".format(self, "Clean", self.path, self.batch_size,
                                                                     self.preprocessing, self.image_shape)
