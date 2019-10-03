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


import onnx
import onnxruntime
import numpy as np

from OpenDenoising.model import module_logger
from OpenDenoising.model import AbstractDeepLearningModel


class OnnxModel(AbstractDeepLearningModel):
    """Onnx models class wrapper. Note that Onnx models only support inference, so training is unavailable.

    Attributes
    ----------
    runtime_session : :class:`onnxruntime.capi.session.InferenceSession`
        onnxruntime session to run inference.
    model_input : :class:`onnxruntime.capi.onnxruntime_pybind11_state.NodeArg`
        onnxruntime input tensor
    model_output : :class:`onnxruntime.capi.onnxruntime_pybind11_state.NodeArg`
        onnxruntime output tensor

    See Also
    --------
    :class:`model.AbstractDenoiser` : for the basic functionalities of Image Denoisers.
    :class:`model.AbstractDeepLearningModel` : for the basic functionalities of Deep Learning based Denoisers.
    """
    def __init__(self, model_name="DeepLearningModel", return_diff=False):
        super().__init__(model_name, None, "Onnx", return_diff=return_diff)
        self.runtime_session = None
        self.model_input = None
        self.model_output = None
        self.channels_first = None

    def charge_model(self, model_path=None):
        """This method charges a onnx model into the class wrapper. It uses onnx module to load the model graph from
        a .onnx file, then creates a runtime session from onnxruntime module.

        Parameters
        ----------
        model_path : str
            String containing the path to the .onnx model file.
        """
        assert (model_path is not None), "You should provide the path for the ONNX model you want to charge"
        self.model = onnx.load(model_path)
        self.runtime_session = onnxruntime.InferenceSession(model_path)
        self.model_input = self.runtime_session.get_inputs()[0]
        self.model_output = self.runtime_session.get_outputs()[0]
        self.channels_first = True if self.model_input.shape[1] in [1, 3] else False


        module_logger.debug("Model inputs and outputs:")
        module_logger.debug("[INPUT] Name: {}, Type: {}, Shape: {}".format(self.model_input.name, self.model_input.type,
                                                                          self.model_input.shape))
        module_logger.debug("[OUTPUT] Name: {}, Type: {}, Shape: {}".format(self.model_output.name,
                                                                           self.model_output.type,
                                                                           self.model_output.shape))

    def train(self, train_generator, valid_generator=None, n_epochs=250, n_stages=500, learning_rate=1e-3,
              metrics=None, optimizer_name=None, kcallbacks=None, loss=None, valid_steps=10):
        raise NotImplementedError("ONNX format does not support training.")

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
        image = image.astype(np.float32)
        if self.channels_first and image.shape[-1] in [1, 3]:
            # Expected channels first, but got channels last
            image = np.transpose(image, [0, 3, 1, 2])
        elif not self.channels_first and image.shape[1] in [1, 3]:
            # Expected channels last, but got channels first
            image = np.transpose(image, [0, 2, 3, 1])
        feed_dict = {self.model_input.name: image}
        predicted_image = self.runtime_session.run([self.model_output.name], feed_dict)[0]
        if self.return_diff:
            return np.clip(image - predicted_image, 0, 1)
        else:
            return np.clip(predicted_image, 0, 1)

    def __len__(self):
        """Counts the number of parameters in the networks.

        Returns
        -------
        nparams : int
            Number of parameters in the network.
        """
        params = self.model.graph.initializer
        nparams = 0
        for param in params:
            nparams += np.prod(param.dims)
        return nparams
