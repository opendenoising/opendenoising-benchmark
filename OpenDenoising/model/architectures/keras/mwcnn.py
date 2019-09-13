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
from keras import layers, models, backend


class DWT(layers.Layer):
    """Custom keras layer implementing the Discrete Wavelet Transform

    Parameters
    ----------
    output_dim : list
        Tuple specifying the output array dimension.
    channels_first : bool
        Whether inputs are formatted as NCHW or NHWC
    """
    def __init__(self, output_dim=None, channels_first=False, **kwargs):
        self.output_dim = output_dim
        self.channels_first = channels_first
        super(DWT, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DWT, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        if self.channels_first:
            x01 = x[:, :, 0::2, :] / 2
            x02 = x[:, :, 1::2, :] / 2
            x1 = x01[:, :, :, 0::2]
            x2 = x02[:, :, :, 0::2]
            x3 = x01[:, :, :, 1::2]
            x4 = x02[:, :, :, 1::2]
            x_ll = x1 + x2 + x3 + x4
            x_hl = -x1 - x2 + x3 + x4
            x_lh = -x1 + x2 - x3 + x4
            x_hh = x1 - x2 - x3 + x4

            return backend.concatenate([x_ll, x_hl, x_lh, x_hh], axis=1)
        else:
            x01 = x[:, 0::2, :, :] / 2
            x02 = x[:, 1::2, :, :] / 2
            x1 = x01[:, :, 0::2, :]
            x2 = x02[:, :, 0::2, :]
            x3 = x01[:, :, 1::2, :]
            x4 = x02[:, :, 1::2, :]
            x_ll = x1 + x2 + x3 + x4
            x_hl = -x1 - x2 + x3 + x4
            x_lh = -x1 + x2 - x3 + x4
            x_hh = x1 - x2 - x3 + x4

            return backend.concatenate([x_ll, x_hl, x_lh, x_hh], axis=-1)

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            return input_shape[0], input_shape[1] // 2, input_shape[2] // 2, input_shape[3] * 4
        else:
            return input_shape[0], None, None, 4 * input_shape[-1]


class IDWT(layers.Layer):
    """Custom keras layer implementing the Inverse Discrete Wavelet Transform

    Parameters
    ----------
    output_dim : list
        Tuple specifying the output array dimension.
    channels_first : bool
        Whether inputs are formatted as NCHW or NHWC
    """
    def __init__(self, output_dim=None, channels_first=False, **kwargs):
        self.channels_first = channels_first
        self.output_dim = output_dim
        super(IDWT, self).__init__(**kwargs)

    def call(self, x):
        assert (len(x.shape.as_list()) == 4), "Expected a 4D input batch, but got {}".format(len(x.shape.as_list()))
        if self.channels_first:
            in_shape = x.shape.as_list()
            out_channel = 4 * in_shape[-1]

            x1 = x[:, 0:out_channel, :, :] / 2
            x2 = x[:, out_channel:out_channel * 2, :, :] / 2
            x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
            x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

            h1 = x1 - x2 - x3 + x4
            h2 = x1 - x2 + x3 - x4
            h3 = x1 + x2 - x3 - x4
            h4 = x1 + x2 + x3 + x4

            h_stack_height = backend.stack([h1, h2], axis=2)
            h_stack_width = backend.stack([h3, h4], axis=2)
            h = backend.stack([h_stack_height, h_stack_width], axis=3)

            return h
        else:
            in_shape = x.shape.as_list()
            out_channel = in_shape[-1] // 4

            x1 = x[:, :, :, 0: out_channel] / 2
            x2 = x[:, :, :, out_channel:out_channel * 2] / 2
            x3 = x[:, :, :, out_channel * 2:out_channel * 3] / 2
            x4 = x[:, :, :, out_channel * 3:out_channel * 4] / 2

            h1 = x1 - x2 - x3 + x4
            h2 = x1 - x2 + x3 - x4
            h3 = x1 + x2 - x3 - x4
            h4 = x1 + x2 + x3 + x4

            h_stack_height = backend.concatenate([h1, h2], axis=1)
            h_stack_width = backend.concatenate([h3, h4], axis=1)
            h = backend.concatenate([h_stack_height, h_stack_width], axis=2)
            return h

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            return input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[3] // 4
        else:
            return input_shape[0], None, None, input_shape[-1] // 4


def mwcnn_1(n_levels=3, kernel_size=3, n_conv_blocks=4, n_channels=1, channels_first=False):
    """Keras implementation of Multilevel Wavelet-CNN from [1].

    Notes
    -----
    The implementation is based on the Pytorch version, available on `this Github page
    <https://github.com/lpj0/MWCNN>`_.

    Parameters
    ----------
    n_levels : int
        Number of Discrete Wavelet Transforms in the network.
    kernel_size : list
        2D Tuple specifying the size of the kernel window used to compute activations.
    n_conv_blocks : int
        Number of convolutional blocks (Conv + BN + ReLU) following each DWT.
    n_channels : int
        Number of image channels that the network processes (1 for grayscale, 3 for RGB)
    channels_first : bool
        Whether channels comes first (NCHW) or last (NHWC)

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
    x = layers.Input([n_channels, None, None]) if channels_first else layers.Input([None, None, n_channels])
    data_format = "channels_first" if channels_first else "channels_last"
    y = x
    filters = [160, 256, 256]
    downsampled_x = []

    """ Downsampling """
    for i in range(n_levels - 1):
        with tf.name_scope("Level_{}_Left".format(i)):
            y = DWT()(y)
            for j in range(n_conv_blocks):
                with tf.name_scope("FullConvNet_{}".format(j)):
                    y = layers.Conv2D(filters=filters[i], kernel_size=kernel_size, strides=(1, 1), padding='same',
                                      kernel_initializer='Orthogonal', use_bias=False, data_format=data_format)(y)
                    y = layers.BatchNormalization(axis=-1, momentum=0.0, epsilon=1e-3)(y)
                    y = layers.Activation("relu")(y)
            downsampled_x.append(y)

    """ Bottom """
    with tf.name_scope("Level_{}".format(n_levels)):
        y = DWT()(y)
        for i in range(2 * n_conv_blocks - 1):
            with tf.name_scope("FullConvNet_{}".format(i)):
                y = layers.Conv2D(filters=256, kernel_size=kernel_size, strides=(1, 1), padding='same',
                                  kernel_initializer='Orthogonal', use_bias=False, data_format=data_format)(y)
                y = layers.BatchNormalization(axis=-1, momentum=0.0, epsilon=1e-3)(y)
                y = layers.Activation("relu")(y)
        with tf.name_scope("FullConvNet_{}".format(2 * n_conv_blocks - 1)):
            y = layers.Conv2D(filters=1024, kernel_size=kernel_size, strides=(1, 1), padding='same',
                              kernel_initializer='Orthogonal', use_bias=False, data_format=data_format)(y)
            y = layers.BatchNormalization(axis=-1, momentum=0.0, epsilon=1e-3)(y)
            y = layers.Activation("relu")(y)

    downsampled_x = downsampled_x[::-1]
    """ Upsampling """
    for i in range(n_levels - 1):
        y = layers.Add()([IDWT()(y), downsampled_x[i]])
        for i in range(n_conv_blocks - 1):
            with tf.name_scope("FullConvNet_{}".format(i)):
                y = layers.Conv2D(filters=256, kernel_size=kernel_size, strides=(1, 1), padding='same',
                                  kernel_initializer='Orthogonal', use_bias=False, data_format=data_format)(y)
                y = layers.BatchNormalization(axis=-1, momentum=0.0, epsilon=1e-3)(y)
                y = layers.Activation("relu")(y)
        with tf.name_scope("FullConvNet_{}".format(n_conv_blocks - 1)):
            y = layers.Conv2D(filters=640, kernel_size=kernel_size, strides=(1, 1), padding='same',
                              kernel_initializer='Orthogonal', use_bias=False, data_format=data_format)(y)
            y = layers.BatchNormalization(axis=-1, momentum=0.0, epsilon=1e-3)(y)
            y = layers.Activation("relu")(y)

    with tf.name_scope("Output"):
        y = layers.Add()([IDWT()(y), x])

    return models.Model(x, y)
