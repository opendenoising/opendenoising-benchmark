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


import numpy as np
import tensorflow as tf
from keras import layers, models, initializers


def rednet(depth=20, n_filters=64, kernel_size=(5, 5), skip_step=2, n_channels=1, channels_first=False):
    """Keras implementation of RedNet. Implementation following the paper [1]_.

    Parameters
    ----------
    depth : int
        Number of fully convolutional layers in dncnn. In the original paper, the authors have used depth=17
        for non-blind denoising and depth=20 for blind denoising.
    n_filters : int
        Number of filters at each convolutional layer.
    kernel_size : list
        2D Tuple specifying the size of the kernel window used to compute activations.
    skip_step : int
        Step for connecting encoder layers with decoder layers through add. For skip_step=2, at each
        2 layers, the j-th encoder layer E_j is connected with the  i = (depth - j) th decoder layer D_i.
    n_channels : int
        Number of image channels that the network processes (1 for grayscale, 3 for RGB)
    channels_first : bool
        Whether channels comes first (NCHW) or last (NHWC)

    Returns
    -------
    :class:`keras.models.Model`
        Keras Model representing the Denoiser neural network

    References
    ----------
    .. [1] Mao XJ, Shen C, Yang YB. Image restoration using convolutional auto-encoders with symmetric skip connections.
           arXiv preprint, 2016.
    """
    num_connections = np.ceil(depth / (2 * skip_step))
    x = layers.Input(shape=[None, None, n_channels], name="InputImage")
    y = x
    encoder_layers = []
    with tf.name_scope("REDNet"):
        for i in range(depth // 2):
            with tf.name_scope("EncoderLayer{}".format(i + 1)):
                y = layers.Conv2D(n_filters, kernel_size=kernel_size,
                                  kernel_initializer=initializers.glorot_uniform(),
                                  padding="same", activation=None, use_bias=False,
                                  name="Layer{}_Conv".format(i + 1))(y)
                y = layers.BatchNormalization(name="Layer{}_BatchNorm".format(i + 1))(y)
                y = layers.ReLU(name="Layer{}_Actv".format(i + 1))(y)
                encoder_layers.append(y)
        j = int((num_connections - 1) * skip_step)  # Encoder layers count
        k = int(depth - (num_connections - 1) * skip_step)  # Decoder layers count
        for i in range(depth // 2 + 1, depth):
            with tf.name_scope("DecoderLayer{}".format(i + 1)):
                y = layers.Conv2DTranspose(n_filters, kernel_size=kernel_size,
                                           kernel_initializer=initializers.glorot_uniform(),
                                           padding="same", activation=None, use_bias=False,
                                           name="Layer{}_Conv".format(i))(y)
                y = layers.BatchNormalization(name="Layer{}_BatchNorm".format(i))(y)
                if i == k:
                    y = layers.Add(name="SkipConnect_Enc_{}_Dec_{}".format(j, k))([encoder_layers[j - 1], y])
                    k += skip_step
                    j -= skip_step
                y = layers.ReLU(name="Layer{}_Actv".format(i))(y)
        with tf.name_scope("OutputLayer"):
            y = layers.Conv2DTranspose(1, kernel_size=kernel_size,
                                       kernel_initializer=initializers.glorot_uniform(),
                                       padding="same", activation=None, use_bias=False,
                                       name="Output_Conv")(y)
            y = layers.BatchNormalization(name="Output_BatchNorm")(y)
            y = layers.Add(name="SkipConnect_Input_Output")([x, y])
            y = layers.ReLU(name="Output_Actv")(y)
    return models.Model(inputs=[x], outputs=[y])


if __name__ == "__main__":
    rednet10 = rednet(10)
    rednet10_json = rednet10.to_json()
    rednet20 = rednet(20)
    rednet20_json = rednet20.to_json()
    rednet30 = rednet(30)
    rednet30_json = rednet30.to_json()

    with open("rednet10.json", 'w') as f:
        f.write(rednet10_json)

    with open("rednet20.json", 'w') as f:
        f.write(rednet20_json)

    with open("rednet30.json", 'w') as f:
        f.write(rednet30_json)

    print("REDNet 10 has {} parameters".format(rednet10.count_params()))
    print("REDNet 20 has {} parameters".format(rednet20.count_params()))
    print("REDNet 30 has {} parameters".format(rednet30.count_params()))
