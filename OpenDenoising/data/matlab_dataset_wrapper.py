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

import os


class MatlabDatasetWrapper:
    """This class wraps FullMatlabDataset and CleanMatlabDataset classes. It makes internal calls to Matlab through
    Python Matlab's engine to load one these classes into the workspace.

    Notes
    -----
    This functions does not implement the interface provided by keras.utils.Sequence. It should only be used alongside
    with MatlabModel objects.

    Attributes
    ----------
    engine : :class:`matlab.engine.MatlabEngine`
        Matlab engine instance.
    images_path : str
        String to directory containing dataset's images.
    partition : str
        String containing the dataset partition ('Train', 'Valid' or 'Test').
    ext : dict
        Dictionary holding images extensions. Note that this dictionary should have keys only.
    patch_size : int
        Size of patches to be extracted.
    n_patches : int
        Number of patches to be extracted on each image.
    noiseFcn : str
        For CleanDataset only. Specifies the noising function that will be applied to images. It should be a function
        that accepts as input an image, and returns another image. If you need to specify parameters, you can do so by
        using Matlab's anonymous function syntax (by specifying noiseFcn="@(I) imnoise(I, 'gaussian', 0, 25/255)"), for
        instance.
    channel_format : str
        String containing 'grayscale' for grayscale images, or 'RGB' for RGB images.
    type : str
        String containing Clean (for CleanDataset) or Full (for FullDataset).

    See Also
    --------
    :class:`model.MatlabModel` : for the type of model for which this class was designed to interact.
    """
    def __init__(self, engine, images_path="./tmp/BSDS500/Train/ref", partition="Train", ext=None, patch_size=40,
                 n_patches=16, noiseFcn="@(I) imnoise(I, 'gaussian', 0, 25/255)", channel_format="grayscale",
                 type="Clean"):
        self.engine = engine
        self.images_path = images_path
        self.ext = {".jpg", ".png"} if ext is None else ext
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.noiseFcn = noiseFcn
        self.channel_format = channel_format
        self.partition = partition
        self.type = type

    def __call__(self):
        if self.type == "Clean":
            self.engine.evalc("imds_{} = imageDatastore('{}', 'FileExtensions', {})".format(self.partition,
                                                                                            self.images_path,
                                                                                            self.ext))
            self.engine.evalc("imds_{}_noise = CleanMatlabDataset(imds_{}, 'PatchSize', {}, " \
                              "'PatchesPerImage', {}, 'noiseFcn', {}," \
                              "'ChannelFormat', '{}')".format(self.partition, self.partition, self.patch_size,
                                                              self.n_patches, self.noiseFcn, self.channel_format))
        elif self.type == "Full":
            self.engine.evalc("imds_{}_in = imageDatastore('{}',"
                              "'FileExtensions', {})".format(os.path.join(self.partition, "in"),
                                                             self.images_path, self.ext))
            self.engine.evalc("imds_{}_ref = imageDatastore('{}',"
                              "'FileExtensions', {})".format(os.path.join(self.partition, "ref"),
                                                             self.images_path, self.ext))
            self.engine.evalc("imds_{}_noise = FullMatlabDataset(imds_{}_in, imds_{}_ref, 'PatchSize', {}, "
                              "'PatchesPerImage', {}, "
                              "'ChannelFormat', '{}')".format(self.partition, self.partition, self.partition,
                                                              self.patch_size, self.n_patches, self.channel_format))
        else:
            raise ValueError("Expected type to be 'Clean' or 'Full', but got {}".format(self.type))
