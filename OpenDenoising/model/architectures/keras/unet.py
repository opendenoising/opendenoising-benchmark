import numpy as np
from keras import layers, models


def unet_CSBDeep(depth=2, n_conv_per_depth=2, n_filters_base=16, kernel_size=(3, 3), n_channels=1,
                 use_bnorm=False, dropout=0.0, pool=(2, 2), residual=True, prob_out=False,
                 eps_scale=1e-3):
    """Noise2void [1]_ uses this U-Net [2]_ network for image denoising. The architecture for denoising is also
    described on [3]_.

    Notes
    -----
    The current code is an adaptation to that present on  `CSBDeep github 
    repository <https://github.com/CSBDeep/CSBDeep>`_, the framework used by the authors of [1]_.

    Parameters
    ----------
    depth : int
        U-net network depth (number of downsampling/upsampling stages).
    n_conv_per_depth : int
        Number of convolutions per network's depth.
    n_filters_base : int
        Number of filters of convolutions on the base depth. At each depth, the number of filters is multiplied
        by two.
    kernel_size : tuple
        Number of pixels on each dimension of convolution kernel.
    n_channels : int
        Number of input image channels (1 for grayscale, 3 for RGB).
    use_bnorm : bool
        Introduces Batch Normalization after convolutions (True) or not.
    dropout : float
        If 0.0, does not introduce dropbout. If greater than 0.0, introduces dropout after activations.
    pool : tuple
        Size of pooling window.
    residual : bool
        Whether to perform residual learning or not (prediction - image_noisy)
    prob_out : bool
        If True, performs probabilistic regression
    eps_scale : float
        Minimum probability for prob_out = True.

    References
    ----------
    .. [1] Krull, Alexander, Tim-Oliver Buchholz, and Florian Jug. "Noise2void-learning denoising from single noisy 
           images." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
    .. [2] Olaf Ronneberger, Philipp Fischer, Thomas Brox, *U-Net: Convolutional Networks for Biomedical Image 
           Segmentation*, MICCAI 2015
    .. [3] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition*, CVPR 2016

    """
    skip_connections = []

    x = layers.Input(shape=(None, None, n_channels))
    y = x

    # Downsampling
    for n in range(depth):
        for l in range(n_conv_per_depth):
            y = layers.Conv2D(filters=n_filters_base * (2 ** n),
                              kernel_size=kernel_size,
                              padding="same",
                              strides=(1, 1),
                              kernel_initializer="glorot_uniform")(y)
            if use_bnorm: y = layers.BatchNormalization()(y)
            y = layers.Activation('relu')(y)
            if dropout > 0: y = layers.Dropout(dropout)(y)
        skip_connections.append(y)
        y = layers.MaxPooling2D(pool)(y)

    # Middle
    for i in range(n_conv_per_depth - 1):
            y = layers.Conv2D(filters=n_filters_base * (2 ** depth),
                              kernel_size=kernel_size,
                              padding="same",
                              strides=(1, 1),
                              kernel_initializer="glorot_uniform")(y)
            if use_bnorm: y = layers.BatchNormalization()(y)
            y = layers.Activation('relu')(y)
            if dropout > 0: y = layers.Dropout(dropout)(y)
    y = layers.Conv2D(filters=n_filters_base * (2 ** max(0, depth - 1)),
                      kernel_size=kernel_size,
                      padding="same",
                      strides=(1, 1),
                      kernel_initializer="glorot_uniform")(y)
    if use_bnorm: y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    if dropout > 0: y = layers.Dropout(dropout)(y)

    # Upsampling
    for n in reversed(range(depth)):
        y = layers.Concatenate(axis=-1)([layers.UpSampling2D(pool)(y), skip_connections[n]])
        for i in range(n_conv_per_depth - 1):
            y = layers.Conv2D(filters=n_filters_base * (2 ** n),
                              kernel_size=kernel_size,
                              padding="same",
                              strides=(1, 1),
                              kernel_initializer="glorot_uniform")(y)
            if use_bnorm: y = layers.BatchNormalization()(y)
            y = layers.Activation('relu')(y)
            if dropout > 0: y = layers.Dropout(dropout)(y)
        y = layers.Conv2D(filters=n_filters_base * (2 ** max(0, n - 1)),
                          kernel_size=kernel_size,
                          padding="same",
                          strides=(1, 1),
                          kernel_initializer="glorot_uniform")(y)
        if use_bnorm: y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        if dropout > 0: y = layers.Dropout(dropout)(y)

    # Final layer (linear activation)
    y = layers.Conv2D(filters=n_channels,
                      kernel_size=kernel_size,
                      padding="same",
                      strides=(1, 1),
                      kernel_initializer="glorot_uniform")(y)
    if residual: # Applies residual learning: pred - input
        y = layers.Add()([y, x])

    if prob_out:
        scale = layers.Conv2D(filters=n_channels,
                              kernel_size=kernel_size,
                              padding="same",
                              strides=(1, 1),
                              kernel_initializer="glorot_uniform",
                              activation="softplus")(y)
        scale = layers.Lambda(lambda x: x + np.float32(eps_scale))(scale)
        y = layers.Concatenate(axis=-1)([y, scale])

    return models.Model(inputs=[x], outputs=[y])
