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


def dncnn(depth=17, n_filters=64, kernel_size=3, n_channels=1, channels_first=False):
    """Tensorflow implementation of DnCNN. Implementation followed the original paper [1]_. Authors original code can be
    found on `their Github Page
    <https://github.com/cszn/DnCNN/>`_.

    Notes
    -----
    Implementation was based on the following `Github page
    <https://github.com/wbhu/DnCNN-tensorflow>`_.

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
    input_tensor : :class:`tf.Tensor`
        Network graph input tensor
    is_training : :class:`tf.Tensor`
        Placeholder indicating if the network is being trained or evaluated
    output_tensor : :class:`tf.Tensor`
        Network graph output tensor

    References
    ----------
    .. [1] Zhang K, Zuo W, Chen Y, Meng D, Zhang L. Beyond a gaussian denoiser: Residual learning of deep cnn
           for image denoising. IEEE Transactions on Image Processing. 2017

    Examples
    --------
    >>> from OpenDenoising.model.architectures.tensorflow import dncnn
    >>> (dncnn_s_input, dncnn_s_is_training, dncnn_s_output) = dncnn(depth=17)
    >>> (dncnn_b_input, dncnn_b_is_training, dncnn_b_output) = dncnn(depth=20)

    """
    assert (n_channels == 1 or n_channels == 3), "Expected 'n_channels' to be 1 or 3, but got {}".format(n_channels)

    if channels_first:
        data_format = "channels_first"
        input_tensor = tf.placeholder(tf.float32, [None, n_channels, None, None], name="input")
    else:
        data_format = "channels_last"
        input_tensor = tf.placeholder(tf.float32, [None, None, None, n_channels], name="input")
    is_training = tf.placeholder(tf.bool, (), name="is_training")

    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(inputs=input_tensor,
                                  filters=n_filters,
                                  kernel_size=kernel_size,
                                  padding='same',
                                  data_format=data_format,
                                  activation=tf.nn.relu)
    for layers in range(2, depth):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(inputs=output,
                                      filters=n_filters,
                                      kernel_size=kernel_size,
                                      padding='same',
                                      name='conv%d' % layers,
                                      data_format=data_format,
                                      use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block{}'.format(depth)):
        noise = tf.layers.conv2d(inputs=output,
                                 filters=n_channels,
                                 kernel_size=kernel_size,
                                 padding='same',
                                 data_format=data_format,
                                 use_bias=False)
    output = tf.subtract(input_tensor, noise, name="output")
    return input_tensor, is_training, output
