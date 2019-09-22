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


import keras
import numpy as np


class AbstractDatasetGenerator(keras.utils.Sequence):
    """Dataset generator based on Keras library. implementation based on
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

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
    """
    def __init__(self, path, batch_size=32, shuffle=True, name="AbstractDataset", n_channels=1):
        self.path = path
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.idx = -1
        self.name = name
        self.filenames = None
        self.idx = None

    def __len__(self):
        """Number of batches per epoch """
        return len(self.filenames) // self.batch_size

    def on_epoch_end(self):
        """Defines and shuffles indexes on epoch end """
        np.random.shuffle(self.filenames)

    def __next__(self):
        """Returns image batches sequentially."""
        self.idx = (self.idx + 1) % len(self)
        return self.__getitem__(self.idx)

    def __str__(self):
        """Returns the dataset name. """
        return self.name
