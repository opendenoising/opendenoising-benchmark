#Copyright or © or Copr. IETR/INSA Rennes (2019)
#
#Contributors :
#    Eduardo Fernandes-Montesuma eduardo.fernandes-montesuma@insa-rennes.fr (2019)
#    Florian Lemarchand florian.lemarchand@insa-rennes.fr (2019)
#
#
#OpenDenoising is a computer program whose purpose is to benchmark image
#restoration algorithms.
#
#This software is governed by the CeCILL-C license under French law and
#abiding by the rules of distribution of free software. You can  use,
#modify and/ or redistribute the software under the terms of the CeCILL-C
#license as circulated by CEA, CNRS and INRIA at the following URL
#"http://www.cecill.info".
#
#As a counterpart to the access to the source code and rights to copy,
#modify and redistribute granted by the license, users are provided only
#with a limited warranty  and the software's author, the holder of the
#economic rights, and the successive licensors have only  limited
#liability.
#
#In this respect, the user's attention is drawn to the risks associated
#with loading, using, modifying and/or developing or reproducing the
#software by the user in light of its specific status of free software,
#that may mean  that it is complicated to manipulate,  and  that  also
#therefore means  that it is reserved for developers  and  experienced
#professionals having in-depth computer knowledge. Users are therefore
#encouraged to load and test the software's suitability as regards their
#requirements in conditions enabling the security of their systems and/or
#data to be ensured and, more generally, to use and operate it in the
#same conditions as regards security.
#
#The fact that you are presently reading this means that you have had
#knowledge of the CeCILL-C license and that you accept its terms.


import tensorflow as tf
from keras import layers, models


def dncnn(depth=17, n_filters=64, kernel_size=(3, 3), n_channels=1, channels_first=False):
    """Keras implementation of DnCNN. Implementation followed the original paper [1]. Authors original code can be
    found on `their Github Page
    <https://github.com/cszn/DnCNN/>`_.

    Parameters
    ----------
    depth : int
        Number of fully convolutional layers in dncnn. In the original paper, the authors have used depth=17 for non-
        blind denoising and depth=20 for blind denoising.
    n_filters : int
        Number of filters on each convolutional layer.
    kernel_size : int tuple
        2D Tuple specifying the size of the kernel window used to compute activations.
    n_channels : int
        Number of image channels that the network processes (1 for grayscale, 3 for RGB)
    channels_first : bool
        Whether channels comes first (NCHW, True) or last (NHWC, False)

    Returns
    -------
    :class:`keras.models.Model`
        Keras model object representing the Neural Network.

    References
    ----------
    .. [1] Zhang K, Zuo W, Chen Y, Meng D, Zhang L. Beyond a gaussian denoiser: Residual learning of deep cnn
           for image denoising. IEEE Transactions on Image Processing. 2017

    Example
    -------
    >>> from OpenDenoising.model.architectures.keras import dncnn
    >>> dncnn_s = dncnn(depth=17)
    >>> dncnn_b = dncnn(depth=20)

    """
    assert (n_channels == 1 or n_channels == 3), "Expected 'n_channels' to be 1 or 3, but got {}".format(n_channels)
    if channels_first:
        data_format = "channels_first"
        x = layers.Input(shape=[n_channels, None, None])
    else:
        data_format = "channels_last"
        x = layers.Input(shape=[None, None, n_channels])
    with tf.name_scope("Layer1"):
        # First layer: Conv + ReLU
        y = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, strides=(1, 1), padding='same',
                          kernel_initializer='Orthogonal', data_format=data_format)(x)
        y = layers.Activation("relu")(y)

    # Middle layers: Conv + ReLU + BN
    for i in range(1, depth - 1):
        with tf.name_scope("Layer{}".format(i + 1)):
            y = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, strides=(1, 1), padding='same',
                              kernel_initializer='Orthogonal', use_bias=False, data_format=data_format)(y)
            y = layers.BatchNormalization(axis=-1, momentum=0.0, epsilon=1e-3)(y)
            y = layers.Activation("relu")(y)

    with tf.name_scope("Layer{}".format(depth)):
        # Final layer: Conv
        y = layers.Conv2D(filters=1, kernel_size=kernel_size, strides=(1, 1), use_bias=False,
                          kernel_initializer='Orthogonal', padding='same', data_format=data_format)(y)
        y = layers.Subtract()([x, y])

    # Keras model
    return models.Model(x, y)
