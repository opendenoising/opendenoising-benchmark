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

from keras import backend
from skimage.measure import compare_mse
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr


class Metric:
    """The metric class is used for interfacing between tensorflow and numpy metrics.

    Notes
    -----
    This class is recommended instead of directly using functions, because functions that work with tensors are not
    recommended to act on numpy arrays (for each time they are called on numpy arrays, a new tensor is created, thus
    causing memory overflow). See Examples for more informations. We remark that, for inference on the benchmark, you
    should always specify np metrics.

    Attributes
    ----------
    tf_metric : function
        Tensorflow function implementing the metric on tensors.
    np_metric : function
        Numpy function implementing metric on ndarrays.

    Examples
    --------
    The most basic usage of Metric class is when you have functions for processing tensors and numpy arrays. We provide
    as built-in metrics SSIM, PSNR and MSE. For instance,

    >>> import numpy as np
    >>> import tensorflow as tf
    >>> from OpenDenoising.evaluation import Metric, tf_ssim, skimage_ssim
    >>> ssim = Metric(name="SSIM", tf_metric=tf_ssim, np_metric=skimage_ssim)
    >>> x = tf.placeholder(tf.float32, [None, None, None, 1])
    >>> y = tf.placeholder(tf.float32, [None, None, None, 1])
    >>> ssim(x, y)
    <tf.Tensor 'Mean_3:0' shape=() dtype=float32>
    >>> x_np = np.random.randn(10, 256, 256, 1)
    >>> y_np = np.random.randn(10, 256, 256, 1)
    >>> ssim(x_np, y_np)
    0.007000506155677978

    That is, if you have specified the two metrics, the class handles if the result is a tensor, or a numeric value.
    """

    def __init__(self, name, tf_metric=None, np_metric=None):
        assert tf_metric is not None or np_metric is not None, "Trying to instantiate metric without functions."
        self.__name__ = name
        self.tf_metric = tf_metric
        self.np_metric = np_metric

    def __call__(self, y_true, y_pred):
        if self.np_metric is None and self.tf_metric is None:
            raise NotImplementedError("Unspecified numpy and tensorflow metrics.")
        if y_true.__class__.__name__ == "ndarray" and y_pred.__class__.__name__ == "ndarray":
            return self.np_metric(y_true, y_pred)
        elif y_true.__class__.__name__ == "Tensor" and y_pred.__class__.__name__ == "Tensor":
            return self.tf_metric(y_true, y_pred)
        elif y_true.__class__.__name__ not in ["ndarray", "Tensor"]:
            raise TypeError("Expected y_true to have type 'ndarray' or 'Tensor' but"
                            "got {}".format(y_true.__class__.__name__))
        elif y_pred.__class__.__name__ not in ["ndarray", "Tensor"]:
            raise TypeError("Expected y_pred to have type 'ndarray' or 'Tensor' but"
                            "got {}".format(y_pred.__class__.__name__))
        elif y_pred.__class__.__name__ != y_true.__class__.__name__:
            raise TypeError("Expected y_pred and y_true to have same type, but got {}"
                            " and {}".format(y_pred.__class__.__name__, y_true.__class__.__name__))

    def __str__(self):
        return self.__name__


def wasserstein_loss(y_true, y_pred):
    """Earth mover loss.

    Parameters
    ----------
    y_true : :class:`tf.Tensor`
        Tensor corresponding to ground-truth images (clean).
    y_pred : :class:`tf.Tensor`
        Tensor corresponding to the Network's prediction.

    Returns
    -------
    :class:`tf.Tensor`
        Tensor corresponding to the evaluated metric.
    """
    loss = backend.mean(y_true * y_pred)
    return loss


def tf_ssim(y_true, y_pred):
    """Structural Similarity Index.

    Parameters
    ----------
    y_true : :class:`tf.Tensor`
        Tensor corresponding to ground-truth images (clean).
    y_pred : :class:`tf.Tensor`
        Tensor corresponding to the Network's prediction.

    Returns
    -------
    :class:`tf.Tensor`
        Tensor corresponding to the evaluated metric.
    """
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def tf_mse(y_true, y_pred):
    """Mean Squared Error.

    .. math::
        MSE = \dfrac{1}{N \\times H \\times W \\times C}\sum_{n=0}^{N}\sum_{i=0}^{H}\sum_{j=0}^{W}\sum_{k=0}^{C}(y_{true}
        (n, i, j, k)-y_{pred}(n, i, j, k))^{2}

    Parameters
    ----------
    y_true : :class:`tf.Tensor`
        Tensor corresponding to ground-truth images (clean).
    y_pred : :class:`tf.Tensor`
        Tensor corresponding to the Network's prediction.

    Returns
    -------
    :class:`tf.Tensor`
        Tensor corresponding to the evaluated metric.
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))


def tf_psnr(y_true, y_pred):
    """Peak Signal to Noise Ratio loss.

    .. math::
        PSNR = \dfrac{10}{N}\sum_{n=0}^{N}log_{10}\\biggr(\dfrac{max(y_{true}(n)^{2})}{MSE(y_{true}(n), y_{pred}(n))}\\biggr)

    Parameters
    ----------
    y_true : :class:`tf.Tensor`
        Tensor corresponding to ground-truth images (clean).
    y_pred : :class:`tf.Tensor`
        Tensor corresponding to the Network's prediction.

    Returns
    -------
    :class:`tf.Tensor`
        Tensor corresponding to the evaluated metric.
    """
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def tf_se(y_true, y_pred):
    """Squared Error loss.

    .. math::
        SE = \sum_{n=0}^{N}\sum_{i=0}^{H}\sum_{j=0}^{W}\sum_{k=0}^{C}(y_{true}(n, i, j, k)-y_{pred}(n, i, j, k))^{2}

    Parameters
    ----------
    y_true : :class:`tf.Tensor`
        Tensor corresponding to ground-truth images (clean).
    y_pred : :class:`tf.Tensor`
        Tensor corresponding to the Network's prediction.

    Returns
    -------
    :class:`tf.Tensor`
        Tensor corresponding to the evaluated metric.
    """
    loss = backend.sum(backend.square(y_pred - y_true)) / 2
    return loss


def skimage_mse(y_true, y_pred):
    """Skimage MSE wrapper.

    Parameters
    ----------
    y_true : :class:`numpy.ndarray`
        4D numpy array containing ground-truth images.
    y_pred : :class:`numpy.ndarray`
        4D numpy array containing the Network's prediction.

    Returns
    -------
    float
        Scalar value of MSE between y_true and y_pred
    """

    return compare_mse(y_true, y_pred)


def skimage_ssim(y_true, y_pred):
    """Skimage SSIM wrapper.

    Parameters
    ----------
    y_true : :class:`numpy.ndarray`
        4D numpy array containing ground-truth images.
    y_pred : :class:`numpy.ndarray`
        4D numpy array containing the Network's prediction.

    Returns
    -------
    float
        Scalar value of SSIM between y_true and y_pred
    """
    assert y_true.shape == y_pred.shape, "Expected y_true and y_pred to have the same shape," \
                                         "but got y_true shape: {} and y_pred shape: {}".format(y_true.shape,
                                                                                                y_pred.shape)
    ssim_vals = []
    for y_true_i, y_pred_i in zip(y_true, y_pred):
        ssim_vals.append(
            compare_ssim(np.squeeze(y_true_i), np.squeeze(y_pred_i), multichannel=False, gaussian_weights=True,
                         use_sample_covariance=False, K1=0.01, K2=0.03, sigma=1.5, data_range=1.0)
        )
    return np.mean(ssim_vals)


def skimage_psnr(y_true, y_pred):
    """Skimage PSNR wrapper.

    Parameters
    ----------
    y_true : :class:`numpy.ndarray`
        4D numpy array containing ground-truth images.
    y_pred : :class:`numpy.ndarray`
        4D numpy array containing the Network's prediction.

    Returns
    -------
    float
        Scalar value of PSNR between y_true and y_pred
    """
    return compare_psnr(y_true, y_pred, data_range=1)


ssim = Metric(name="SSIM", tf_metric=tf_ssim, np_metric=skimage_ssim)
psnr = Metric(name="PSNR", tf_metric=tf_psnr, np_metric=skimage_psnr)
mse = Metric(name="MSE", tf_metric=tf_mse, np_metric=skimage_mse)
