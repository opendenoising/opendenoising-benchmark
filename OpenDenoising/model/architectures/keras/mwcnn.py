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


import tensorflow as tf
from keras import layers, models, backend


def DWT(x):
    dwt_hor = layers.Lambda(lambda x: [x[:, 0::2, :, :] / 2, x[:, 1::2, :, :] / 2])
    dwt_ver = layers.Lambda(lambda x: [x[0][:, :, 0::2, :], x[1][:, :, 0::2, :], x[0][:, :, 1::2, :], x[1][:, :, 1::2, :]])
    dwt_precat = layers.Lambda(lambda x:[x[0] + x[1] + x[2] + x[3],
                                         -x[0] - x[1] + x[2] + x[3],
                                         -x[0] + x[1] - x[2] + x[3],
                                         x[0] - x[1] - x[2] + x[3]])

    y = dwt_hor(x)
    y = dwt_ver(y)
    y = dwt_precat(y)

    return layers.Concatenate(axis=-1)(y)


def IDWT(x):
    shape = x.shape.as_list()
    out_channel = shape[-1] // 4
    
    idwt1 = layers.Lambda(lambda x: [x[:, :, :, 0 * out_channel: 1 * out_channel] / 2,
                                     x[:, :, :, 1 * out_channel: 2 * out_channel] / 2,
                                     x[:, :, :, 2 * out_channel: 3 * out_channel] / 2,
                                     x[:, :, :, 3 * out_channel: 4 * out_channel] / 2])
    
    idwt2 = layers.Lambda(lambda x: [x[0] + x[1] + x[2] + x[3],
                                     x[0] - x[1] + x[2] - x[3],
                                     x[0] + x[1] - x[2] + x[3],
                                     x[0] + x[1] + x[2] + x[3]])

    decomposition_1 = idwt1(x)
    decomposition_2 = idwt2(decomposition_1)

    hor_cat_1 = layers.Concatenate(axis=1)([decomposition_2[0], decomposition_2[1]])
    hor_cat_2 = layers.Concatenate(axis=1)([decomposition_2[2], decomposition_2[3]])
    
    return layers.Concatenate(axis=2)([hor_cat_1, hor_cat_2])


def mwcnn(kernel_size=3, n_conv_blocks=4, n_channels=1):
    """Keras implementation of Multilevel Wavelet-CNN from [1].

    Notes
    -----
    The implementation is based on the Pytorch version, available on `this Github page
    <https://github.com/lpj0/MWCNN>`_.

    Parameters
    ----------
    kernel_size : list
        2D Tuple specifying the size of the kernel window used to compute activations.
    n_conv_blocks : int
        Number of convolutional blocks (Conv + BN + ReLU) following each DWT.
    n_channels : int
        Number of image channels that the network processes (1 for grayscale, 3 for RGB)

    Returns
    -------
    :class:`keras.models.Model`
        Keras model object.

    References
    ----------
    .. [1] Liu P, Zhang H, Zhang K, Lin L, Zuo W. Multi-level wavelet-CNN for image restoration. InProceedings of the
           IEEE Conference on Computer Vision and Pattern Recognition Workshops 2018
    """
    assert (n_channels == 1 or n_channels == 3), "Expected 'n_channels' to be 1 or 3, but got {}".format(n_channels)
    x = layers.Input([None, None, n_channels])

    # MWCNN network
    # First DWT
    y1 = DWT()(x)  # has shape [None, None, None, 4]
    for _ in range(4):
        y1 = layers.Conv2D(filters=160, kernel_size=kernel_size, padding='same', use_bias=False)(y1)
        y1 = layers.BatchNormalization()(y1)
        y1 = layers.Activation('relu')(y1)
    # Second DWT
    y2 = DWT()(y1)  # y1 => [None, None, None, 160], y2 => [None, None, None, 640]
    for _ in range(4):
        y2 = layers.Conv2D(filters=256, kernel_size=kernel_size, padding='same', use_bias=False)(y2)
        y2 = layers.BatchNormalization()(y2)
        y2 = layers.Activation('relu')(y2)
    # Third DWT
    y3 = DWT()(y2)  # y2 => [None, None, None, 256], y3 => [None, None, None, 1024]
    for _ in range(7):
        y3 = layers.Conv2D(filters=256, kernel_size=kernel_size, padding='same', use_bias=False)(y3)
        y3 = layers.BatchNormalization()(y3)
        y3 = layers.Activation('relu')(y3)
    y3 = layers.Conv2D(filters=1024, kernel_size=kernel_size, padding='same', use_bias=False)(y3)
    y3 = layers.BatchNormalization()(y3)
    y3 = layers.Activation('relu')(y3)

    # First IDWT
    iy3 = layers.Add()([IDWT()(y3), y2])  # y2 => [None, None, None, 256], iy3 => [None, None, None, 256]
    for _ in range(3):
        iy3 = layers.Conv2D(filters=256, kernel_size=kernel_size, padding='same', use_bias=False)(iy3)
        iy3 = layers.BatchNormalization()(iy3)
        iy3 = layers.Activation('relu')(iy3)
    iy3 = layers.Conv2D(filters=640, kernel_size=kernel_size, padding='same', use_bias=False)(iy3)
    iy3 = layers.BatchNormalization()(iy3)
    iy3 = layers.Activation('relu')(iy3)

    # Second IDWT
    iy2 = layers.Add()([IDWT()(iy3), y1])  # y2 => [None, None, None, 640], y3 => [None, None, None, 640]
    for _ in range(3):
        iy2 = layers.Conv2D(filters=160, kernel_size=kernel_size, padding='same', use_bias=False)(iy2)
        iy2 = layers.BatchNormalization()(iy2)
        iy2 = layers.Activation('relu')(iy2)
    iy2 = layers.Conv2D(filters=4, kernel_size=kernel_size, padding='same', use_bias=False)(iy2)
    iy2 = layers.BatchNormalization()(iy2)
    iy2 = layers.Activation('relu')(iy2)

    # Final IDWT
    y = layers.Add()([IDWT()(iy2), x])

    return models.Model(x, y)
