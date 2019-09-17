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


from keras import layers, models, backend


def xnet(depth=8, n_filters=64, kernel_size=(3, 3), skernel_size=(9, 9), n_channels=1, channels_first=False):
    """xNet implementation on Keras. Implementation followed the paper [1]_.

    Notes
    -----
    The implementation is based on the Pytorch version, available on `this Github page
    <https://github.com/kligvasser/xUnit>`_.

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
    .. [1] Kligvasser I, Rott Shaham T, Michaeli T. xUnit: Learning a spatial activation function for efficient image
           restoration. InProceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2018
    """
    x = layers.Input(shape=[None, None, n_channels])
    y = x
    for _ in range(depth):
        z = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, padding="same", use_bias=False)(y)
        z = layers.BatchNormalization()(z)
        z = layers.Activation('relu')(z)
        z = layers.DepthwiseConv2D(kernel_size=skernel_size, padding="same", use_bias=False)(z)
        d = layers.BatchNormalization()(z)
        g = layers.Lambda(lambda x: backend.exp(- backend.square(x)))(d)

        y = layers.Multiply()([y, g])
    y = layers.Conv2D(filters=1, kernel_size=kernel_size, padding="same")(y)
    return models.Model(x, y)
