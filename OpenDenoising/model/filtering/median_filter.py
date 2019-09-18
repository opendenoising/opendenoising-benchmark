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


import numpy as np
import tensorflow as tf


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def median_filter(image, filter_size=(3, 3), stride=(1, 1)):
    """Naive implementation of median filter using numpy.

    Parameters
    ----------
    image : :class:`numpy.ndarray`
        4D batch of noised images. It has shape: (batch_size, height, width, channels).
    filter_size : list
        2D list containing the size of filter's kernel.
    stride : list
        2D list containing the horizontal and vertical strides.

    Returns
    -------
    output : :class:`numpy.ndarray`
        4D batch of denoised images. It has shape: (batch_size, height, width, channels).
    """
    if image.ndim == 2:
        image = np.expand_dims(np.expand_dims(image, axis=0), axis=-1)
    p = filter_size[0] // 2
    sh, sw = stride

    _image = np.pad(image, pad_width=((0, 0), (p, p), (p, p), (0, 0)), mode="constant")
    N, h, w, c = _image.shape
    output = np.zeros(image.shape)

    for i in range(p, h - p, sh):
        # Loops over horizontal axis
        for j in range(p, w - p, sw):
            # Loops over vertical axis
            window = _image[:, i - p: i + p, j - p: j + p, :]
            output[:, i - p, j - p, :] = np.median(window, axis=(1, 2))

    return output
