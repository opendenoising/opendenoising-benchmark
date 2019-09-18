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
import numpy as np
from OpenDenoising.model import module_logger
from OpenDenoising.model import AbstractDeepLearningModel

try:
    import matlab.engine
    MATLAB_IMPORTED = True
except ImportError as err:
    module_logger.warning("Matlab engine was not installed correctly. Take a look on the documentation's tutorial for \
                          its installation.")
    MATLAB_IMPORTED = False


class MatconvnetModel(AbstractDeepLearningModel):
    """Matlab Matconvnet wrapper class.

    Notes
    -----
    Matconvnet models are available only for inference.

    Attributes
    ----------
    model_path : str
        String containing the path to model files.
    denoising_func : :class:`function`
        Function object representing the Matlab function responsable for denoising images.
    model_file_path : str
        String containing the path to the model saved weights.

    See Also
    --------
    :class:`model.AbstractDenoiser` : for the basic functionalities of Image Denoisers.
    :class:`model.AbstractDeepLearningModel` : for the basic functionalities of Deep Learning based Denoisers.
    """
    def __init__(self, model_name="MatconvnetModel", return_diff=False):
        global MATLAB_IMPORTED
        assert MATLAB_IMPORTED, "Got expcetion {} while importing matlab.engine. Check Matlab's Engine installation.".format(err)

        try:
            self.engine = matlab.engine.start_matlab()
        except matlab.engine.EngineError as err:
            module_logger.exception("Matlab license error. Make sure you have a valid Matlab license.")
            raise err

        super().__init__(model_name, framework="Matconvnet", return_diff=return_diff)
        self.model_path = None
        self.denoise_func = None

    def charge_model(self, model_path=None):
        """This method charges a matlab function corresponding to the denoising action into the class wrapper.

        Parameters
        ----------
        model_path : str
            String containing the path to the Matlab denoising function, as well as the networks weights (saved as a
            .mat file)
        """
        assert (model_path is not None), "Unspecified model_path."
        self.model_path = model_path
        model_dir = os.path.dirname(self.model_path)
        # Adds entire folder to Matlab path
        self.engine.addpath(self.engine.fullfile(model_dir))
        # Lists files in model directory
        files = os.listdir(model_dir)
        # Search for .m and .mat files in model_dir.
        denoise_func_name = None
        for file in files:
            if "denoise" in file:
                # Founded denoise function
                denoise_func_name = file.split(".m")[0]
        assert (denoise_func_name is not None), "Expected 'denoise' to be substring in at least one file in {},"\
                                                "but found files: {}".format(model_dir, self.model_path)
        self.denoise_func = getattr(self.engine, denoise_func_name)

    def train(self, train_generator, valid_generator=None, n_epochs=100, n_stages=5e+2,
              learning_rate=1e-3, optimizer_name="Adam", valid_steps=10):
        raise NotImplementedError("Training of matconvnet-based is not supported.")

    def __call__(self, image):
        """Denoises a batch of images.

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            4D batch of noised images. It has shape: (batch_size, height, width, channels)

        Returns
        -------
        :class:`numpy.ndarray`:
            Restored batch of images, with same shape as the input.

        """
        predicted_image = []

        for img in image:
            img = np.squeeze(img)  # If image is grayscale, converts 3D array to 2D array.
            m_img = matlab.single(img.tolist())  # Converts array to matlab
            m_img_res = self.denoise_func(self.model_path, m_img, matlab.logical([True]))
            predicted_image.append(np.asarray(m_img_res))
        predicted_image = np.asarray(predicted_image)
        if self.return_diff:
            return image - predicted_image
        else:
            return predicted_image

    def __len__(self):
        """Counts the number of parameters in the networks.

        Returns
        -------
        nparams : int
            Number of parameters in the network.
        """
        nparams = self.engine.vl_count_params(self.model_path)
        return int(np.asarray(nparams))
