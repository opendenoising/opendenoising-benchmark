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


import cv2
import numpy as np

from skimage.util.noise import random_noise

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)


def gaussian_noise(ref, noise_level=15):
    """Gaussian noise

    Parameters
    ----------
    ref : :class:`numpy.ndarray`
        Image to be noised.
    noise_level : :class:`numpy.ndarray`
        Level of corruption. Always give the noise_level in terms of 0-255 pixel intensity range.

    Returns
    -------
    inp : :class:`numpy.ndarray`
        Noised image.
    """
    inp = random_noise(ref, mode='gaussian', clip=False, var=(noise_level / 255) ** 2)

    assert inp.shape == ref.shape, "Shape mismatch between input ({}) and output ({})".format(inp.shape, ref.shape)
    return inp


def poisson_noise(ref):
    """Poisson noise

    Parameters
    ----------
    ref : :class:`numpy.ndarray`
        Image to be noised.

    Returns
    -------
    inp : :class:`numpy.ndarray`
        Noised image.
    """
    inp = random_noise(ref, mode='poisson', seed=RANDOM_SEED, clip=True)

    assert inp.shape == ref.shape, "Shape mismatch between input ({}) and output ({})".format(inp.shape, ref.shape)
    return inp


def salt_and_pepper_noise(ref, noise_level=15):
    """Salt and pepper noise

    Parameters
    ----------
    ref : :class:`numpy.ndarray`
        Image to be noised.
    noise_level : :class:`numpy.ndarray`
        Percentage of perturbed pixels.

    Returns
    -------
    inp : :class:`numpy.ndarray`
        Noised image.
    """
    assert noise_level >= 0 and noise_level <= 100, "Expected noise_level to be a percentage," \
                                                    "but got {}".format(noise_level)
    inp = random_noise(ref, mode='s&p', seed=RANDOM_SEED, clip=True, amount=noise_level / 100.0)

    assert inp.shape == ref.shape, "Shape mismatch between input ({}) and output ({})".format(inp.shape, ref.shape)
    return inp


def speckle_noise(ref, noise_level=15):
    """Speckle noise

    Parameters
    ----------
    ref : :class:`numpy.ndarray`
        Image to be noised.
    noise_level : :class:`numpy.ndarray`
        Percentage of perturbed pixels.

    Returns
    -------
    inp : :class:`numpy.ndarray`
        Noised image.
    """
    inp = random_noise(ref, mode='speckle', clip=True, var=noise_level)

    assert inp.shape == ref.shape, "Shape mismatch between input ({}) and output ({})".format(inp.shape, ref.shape)
    return inp


def super_resolution_noise(ref, noise_level=2):
    """Noise due to down-sampling followed by up-sampling an image

    Parameters
    ----------
    ref : :class:`numpy.ndarray`
        Image to be noised.
    noise_level : :class:`numpy.ndarray`
        scaling factor. For instance, for an image (512, 512), a factor 2 down-samples the image to (256, 256), then
        up-samples it again to (512, 512).

    Returns
    -------
    inp : :class:`numpy.ndarray`
        Noised image.
    """
    h, w = ref.shape[:2]
    downsampled_image = cv2.resize(ref, (h // noise_level, w // noise_level), interpolation=cv2.INTER_CUBIC)
    inp = cv2.resize(downsampled_image, (h, w), interpolation=cv2.INTER_CUBIC)

    assert inp.shape == ref.shape, "Shape mismatch between input ({}) and output ({})".format(inp.shape, ref.shape)
    return inp


def gaussian_blind_noise(ref, noise_min=0, noise_max=55):
    """Corruption for Blind Denoising adopted on DnCNN paper.

    Parameters
    ----------
    ref : :class:`numpy.ndarray`
        Image to be noised.
    noise_min : float
        Minimum value for :math:`\sigma` parameter of gaussian noise. It should be specified as if images had range
        0-255.
    noise_max : float
        Maximum value for :math:`\sigma` parameter of gaussian noise. It should be specified as if images had range
        0-255.

    Returns
    -------
    inp : :class:`numpy.ndarray`
        Noised image.
    """
    sigma = np.random.uniform(noise_min, noise_max)
    return gaussian_noise(ref, noise_level=sigma)


def jpeg_artifacts(ref, compression_rate=50):
    """Introduce JPEG artifacts through JPEG compression.

    Parameters
    ----------
    ref : :class:`numpy.ndarray`
        Image to be noised. ref should be a 3D array (height, width, channel) of float dtype (range 0-1).
    compression_rate : int
        Compression percentage. Should be an integer between 0 and 100.
    """
    assert (ref.dtype in ['float32', 'float64']), "Expected reference array to be float32 or float64," \
                                                  "but got {}".format(ref.dtype)
    assert (0 < compression_rate < 100), "Expected compression_rate to be between 0 and 100," \
                                         " but got {}".format(compression_rate)
    # Converts image to uint8
    _img = (ref * 255).astype('uint8')
    # Performs compression
    result, encimg = cv2.imencode('.jpg', _img, [int(cv2.IMWRITE_JPEG_QUALITY), 100 - compression_rate])
    # Decodes image
    decimg = cv2.imdecode(encimg, 1)
    # Converts decimg to float
    decimg = decimg.astype(ref.dtype) / 255

    if ref.ndim == 2 or ref.shape[-1] == 1:
        decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2GRAY)

    return decimg
