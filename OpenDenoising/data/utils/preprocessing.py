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
from OpenDenoising.data import module_logger


def clip_batch(inp, ref=None, minrange=0, maxrange=1):
    """Clips a batch of images, so that all pixels with values less than minrange are set to minrange, and all above
    maxrange at set to maxrange.

    Parameters
    ----------
    inp : :class:`numpy.ndarray`
        4D Batch of images with shape (batch_size, height, width, channels)
    ref : :class:`numpy.ndarray`
        4D Batch of images with shape (batch_size, height, width, channels)
    minrange : float
        Lower bound of pixel interval.
    maxrange : float
        Upper bound of pixel interval.
    """
    if ref is not None:
        return np.clip(inp, minrange, maxrange), np.clip(ref, minrange, maxrange)
    else:
        return np.clip(inp, minrange, maxrange)


def normalize_batch(inp, ref=None, minrange=0, maxrange=1):
    """Normalizes each pixel of a 4D batch of images in the range [minrange, maxrange].

    Parameters
    ----------
    inp : :class:`numpy.ndarray`
        4D Batch of images with shape (batch_size, height, width, channels)
    minrange : float
        Lower bound of pixel interval.
    maxrange : float
        Upper bound of pixel interval.
    """
    n = len(inp)
    _inp = np.zeros(inp.shape)
    _ref = np.zeros(inp.shape) if ref is not None else None

    for i in range(n):
        _inp[i] = (inp[i] - np.amin(inp[i])) / (np.amax(inp[i]) - np.amin(inp[i]))
        if ref is not None:
            _ref[i] = (ref[i] - np.amin(ref[i])) / (np.amax(ref[i]) - np.amin(ref[i]))

    if ref is not None:
        return (maxrange - minrange) * _inp + minrange, (maxrange - minrange) * _ref + minrange
    else:
        return (maxrange - minrange) * _inp + minrange


def dncnn_augmentation(inp, ref=None, aug_times=1, channels_first=False):
    """Data augmentation policy employed on DnCNN

    Parameters
    ----------
    inp : :class:`numpy.ndarray`
        Noised image.
    ref : :class:`numpy.ndarray`
        Ground-truth images.
    aug_times : int
        Number of times augmentation if applied.
    channels_first : boo
        Whether data is formatted as NCHW (True) or NHWC (False).

    Returns
    -------
    inp : :class:`numpy.ndarray`
        Augmented noised images
    ref : :class:`numpy.ndarray`
        Augmented ground-truth images
    """
    _inp = inp.copy()
    _ref = ref.copy if ref is not None else None
    assert (inp.ndim == 4), "Expected 4D input, but got {}D".format(inp.ndim)
    if ref is not None:
        assert (ref.ndim == 4), "Expected 4D reference, but got {}D".format(ref.ndim)
    inp_aug = None
    ref_aug = None
    axes = (2, 3) if channels_first else (1, 2)
    for _ in range(aug_times):
        mode = np.random.randint(0, 7)
        if mode == 0:
            # No augmentation
            inp_aug = inp.copy()
            if ref is not None:
                ref_aug = ref.copy()
        elif mode == 1:
            # Flip image over the x axis (up-down)
            inp_aug = np.flipud(inp)
            if ref is not None:
                ref_aug = np.flipud(ref)
        elif mode == 2:
            # Rotate the image by 90 degrees
            inp_aug = np.rot90(inp, 1, axes=axes)
            if ref is not None:
                ref_aug = np.rot90(ref, 1, axes=axes)
        elif mode == 3:
            # Flip the image over the x axis, then rotate it by 90 degrees
            inp_aug = np.flipud(np.rot90(inp, 1, axes=axes))
            if ref is not None:
                ref_aug = np.flipud(np.rot90(ref, 1, axes=axes))
        elif mode == 4:
            # Rotate the image by 180 degrees
            inp_aug = np.flipud(np.rot90(inp, 1, axes=axes))
            if ref is not None:
                ref_aug = np.flipud(np.rot90(ref, 1, axes=axes))
        elif mode == 5:
            # Flip the image over the x axis, then rotate it by 180 degrees
            inp_aug = np.flipud(np.rot90(inp, 2, axes=axes))
            if ref is not None:
                ref_aug = np.flipud(np.rot90(ref, 2, axes=axes))
        elif mode == 6:
            # Rotate the image by 270 degrees
            inp_aug = np.rot90(inp, 3, axes=axes)
            if ref is not None:
                ref_aug = np.rot90(ref, 3, axes=axes)
        elif mode == 7:
            # Flip the image over the x axis, then rotate it by 270 degrees
            inp_aug = np.flipud(np.rot90(inp, 3, axes=axes))
            if ref is not None:
                ref_aug = np.flipud(np.rot90(ref, 3, axes=axes))
        if inp_aug is not None:
            # Augmentation was performed
            inp = np.concatenate([inp, inp_aug], axis=0)
        if ref is not None and ref_aug is not None:
            # Reference was passed
            ref = np.concatenate([ref, ref_aug], axis=0)
    if ref is not None:
        # Passed input and reference images
        assert inp.shape[1:] == _inp.shape[1:], "Data Augmentation changed input shape: " \
                                                "before {}, after {}".format(_inp.shape[1:], inp.shape[1:])
        assert ref.shape[1:] == ref.shape[1:], "Data Augmentation changed reference shape: " \
                                                "before {}, after {}".format(_inp.shape[1:], inp.shape[1:])
        return inp, ref
    else:
        # Passed only input image
        assert inp.shape[1:] == _inp.shape[1:], "Data Augmentation changed input shape: " \
                                                "before {}, after {}".format(_inp.shape[1:], inp.shape[1:])
        return inp


def gen_patches(inp, ref, patch_size, channels_first=False, mode="sequential", n_patches=-1):
    """Patch generation function.

    Parameters
    ----------
    inp : :class:`numpy.ndarray`
        Noised image which patches will be extracted.
    ref : :class:`numpy.ndarray`
        Reference image which patches will be extracted.
    patch_size : int
        Size of patch window (number of pixels in each axis).
    channels_first : bool
        Whether data is formatted as NCHW (True) or NHWC (False).
    mode : str
        One between {'sequential', 'random'}. If mode = 'sequential', extracts patches sequentially on each axis.
        If mode = 'random', extracts patches randomly.
    n_patches : int
        Number of patches to be extracted from the image. Should be specified only if mode = 'random'. If not specified,
        then,

        .. math::

            n\_patches = \dfrac{h \\times w}{patch_{size}^{2}}

    Returns
    -------
    input_patches : :class:`numpy.ndarray`
        Extracted input patches.
    reference_patches : :class:`numpy.ndarray`
        Extracted reference patches.
    """
    assert mode in ["random", "sequential"], "Expected mode to be 'random' or 'sequential' but got {}".format(mode)

    _inp = inp.copy()
    _ref = ref.copy()
    if channels_first:
        _, h, w = _inp.shape
    else:
        h, w = _inp.shape[:2]

    if mode == "random" and n_patches == -1:
        n_patches = h * w / (patch_size ** 2)

    inp_patches = []
    ref_patches = []

    if mode == "sequential":
        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                if channels_first:
                    inp_patches.append(_inp[:, i: i + patch_size, j: j + patch_size])
                    ref_patches.append(_ref[:, i: i + patch_size, j: j + patch_size])
                else:
                    inp_patches.append(_inp[i: i + patch_size, j: j + patch_size, :])
                    ref_patches.append(_ref[i: i + patch_size, j: j + patch_size, :])
    if mode == "random":
        for _ in range(n_patches):
            i = np.random.randint(0, h - patch_size + 1)
            j = np.random.randint(0, h - patch_size + 1)
            if channels_first:
                inp_patches.append(_inp[:, i: i + patch_size, j: j + patch_size])
                ref_patches.append(_ref[:, i: i + patch_size, j: j + patch_size])
            else:
                inp_patches.append(_inp[i: i + patch_size, j: j + patch_size, :])
                ref_patches.append(_ref[i: i + patch_size, j: j + patch_size, :])


    input_patches = np.array(inp_patches)
    reference_patches = np.array(ref_patches)

    return input_patches, reference_patches


def smooth_patches(img, d=64, h=32, sg=32, sl=16, mu=0.1, gamma=0.25):
    """Extract smooth patches for GCBD algorithm.

    Parameters
    ----------
    img : :class:`numpy.ndarray`
        Noised image.
    d : int
        Global patch size.
    h : int
        Local patch size.
    sg : int
        Global stride.
    sl : int
        Local stride.
    mu : float
        mean-smoothing hyper-parameter.
    gamma : float
        Variance-smoothing hyper-parameter.

    Returns
    -------
    patches : :class:`numpy.ndarray`
        Extracted patches from img.
    """
    height, width = img.shape
    patches = []
    for i in range(0, height - d, sg):
        for j in range(0, width - d, sg):
            # Run for global h x h patches p
            wg = img[i: i + d, j: j + d]
            # Patch mean and variance
            mean_g = np.mean(wg)
            var_g = np.var(wg)
            # Initializes smooth verifier
            smooth = True
            for k in range(i, i + d - h, sl):
                for l in range(j, j + d - h, sl):
                    # Run for local h x h patches q
                    wl = img[k: k + h, l: l + h]
                    module_logger.debug("image shape {}, {}:{}, {}:{}, local patch shape: {}".format(img.shape, k, k + h, l,
                                                                                               l + h, wl.shape))
                    # Local mean and variance
                    mean_l = np.mean(wl)
                    var_l = np.var(wl)
                    # Difference between local and global means/variances
                    mean_diff = np.abs(mean_g - mean_l)
                    var_diff = np.abs(var_g - var_l)
                    if mean_diff > mu * mean_l or var_diff > gamma * var_l:
                        module_logger.debug("{}\tmean_g: {}, mean_l: {}, mean_diff: {}, bound: {}".format([i, j, k, l],
                                                                                                    mean_g,
                                                                                                    mean_l,
                                                                                                    mean_diff,
                                                                                                    mu * mean_l))
                        module_logger.debug("{}\tvar_g: {}, var_l: {}, var_diff: {}, bound: {}\n".format([i, j, k, l], var_g,
                                                                                                   var_l, var_diff,
                                                                                                   gamma * var_l))
                        # If constraints not met for a local patch, then global patch
                        # is not smooth => smooth verifier becomes false
                        smooth = False
            if smooth:
                # If smooth enough, then extracts the noise patch through
                # noise = patch - mean(patch)
                patches.append(wg - np.mean(wg))
    return patches
