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
    from matlab.engine import EngineError
    from matlab.engine import MatlabExecutionError
    MATLAB_IMPORTED = True
except ImportError as err:
    module_logger.warning("Matlab engine was not installed correctly. Take a look on the documentation's tutorial for \
                          its installation.")
    MATLAB_IMPORTED = False


def kwargs_to_string(**kwargs):
    string = ""
    for i, argname in enumerate(kwargs):
        if i < len(kwargs) - 1:
            string += "'{}', {},".format(argname, kwargs[argname])
        else:
            string += "'{}', {}".format(argname, kwargs[argname])
    return string


class MatlabModel(AbstractDeepLearningModel):
    """Matlab Deep Learning toolbox wrapper class.

    Notes
    -----
    To use this class you need a Matlab license with access to the Deep Learning toolbox.

    Attributes
    ----------
    logdir : str
        Path to the directory where training logs will be saved.

    See Also
    --------
    :class:`model.AbstractDenoiser` : for the basic functionalities of Image Denoisers.
    :class:`model.AbstractDeepLearningModel` : for the basic functionalities of Deep Learning based Denoisers.
    """
    def __init__(self, model_name="MatlabModel", logdir="./logs/Matlab", return_diff=False):
        global MATLAB_IMPORTED
        assert MATLAB_IMPORTED, "Got expcetion {} while importing matlab.engine. Check Matlab's Engine installation.".format(err)
        try:
            self.engine = matlab.engine.start_matlab()
        except EngineError as err:
            module_logger.exception("Matlab license error. Make sure you have a valid Matlab license.")
            raise err
        assert self.engine.license('test', 'neural_network_toolbox'), "Expected Neural Network Toolbox to be installed."
        super().__init__(model_name, framework="Matlab", return_diff=return_diff)
        self.logdir = logdir

    def charge_model(self, model_function=None, model_path=None, **kwargs):
        """MatlabModel model charging functions

        This function works by using Matlab's Python engine to make Matlab internal calls. You can either specify the
        path to a Matlab function that will build the model, or to a .mat file holding the pretrained model.

        Notes
        -----
        You may specify optional arguments to the model building function through keyword arguments.

        Parameters
        ----------
        model_function : str
            String containing the path to the matlab .m function that will build the model. Note that this function
            should return a list of layers.
        model_path : str
            String containing the path to the .mat file holding the trained network object.

        """
        assert (model_function is not None or model_path is not None), "You should provide at least a model_function\
                                                                        or a model_path to build your neural network\
                                                                        model"
        if model_function is not None:
            model_function = model_function.split("/")
            model_function_path = "/".join(model_function[:-1])
            function_filename = model_function[-1]
            function_name, _ = os.path.splitext(function_filename)
            self.engine.addpath(os.path.abspath(model_function_path))
            if self.engine.which(function_filename) == '':
                raise ValueError("Function {} is not in PATH".format(model_function))
            self.engine.evalc("layers = {}({});".format(function_name, kwargs_to_string(**kwargs)))
        elif model_path is not None:
            self.engine.evalc("load('{}');".format(model_path))

    def train(self, train_generator, valid_generator=None, n_epochs=250, n_stages=500, learning_rate=1e-3,
              optimizer_name="adam", valid_steps=10):
        """Trains a Matlab denoiser model.

        Notes
        -----
        Instead of using Clean/Full/Blind dataset Python classes, you should use the class MatlabDatasetWrapper,
        which exports a dataset to matlab workspace.

        Parameters
        ----------
        train_generator : str
            Name of the matlab imageDatastore variable (for instance, if you have train_imds in your workspace you
            should pass 'train_imds' for train_generator).
        valid_generator : str
            Name of the matlab imageDatastore variable (for instance, if you have train_imds in your workspace you
            should pass 'valid_imds' for valid_generator).
        n_epochs : int
            Number of training epochs.
        n_stages : int
            Number of image batches are drawn during a training epoch.
        learning_rate : float
            Initial value for learning rate value for optimization.
        optimizer_name : str
            One among {'sgdm', 'adam', 'rmsprop'}

        See Also
        --------
        :class:`data.MatlabDatasetWrapper` : for the class providing data to train such kinds of models.
        """
        assert optimizer_name in ['sgdm', 'adam', 'rmsprop'], "Expected optimizer_name to be in {}, but got {}".format(
            ['sgdm', 'adam', 'rmsprop'], optimizer_name
        )
        if optimizer_name == None:
            optimizer_name = "adam"
        try:
            self.engine.workspace.__getitem__("{}".format(train_generator))
        except MatlabExecutionError as err:
            raise MatlabExecutionError("Variable {} is not in Matlab's workspace."\
                                       "Workspace:\n".format(train_generator, self.engine.workspace))
        if valid_generator:
            print("opts = trainingOptions('{}', {});".format(optimizer_name, kwargs_to_string(
                Plots="'training-progress'", Verbose="true", VerboseFrequency=n_stages,
                MaxEpochs=n_epochs, Shuffle="'every-epoch'", ValidationData="valid_imds",
                CheckpointPath="'{}'".format(self.logdir)
            )))
            # Adds datasets to workspace

            self.engine.evalc("opts = trainingOptions('{}', {});".format(optimizer_name, kwargs_to_string(
                Plots="'training-progress'", Verbose="true", VerboseFrequency=n_stages,
                MaxEpochs=n_epochs, Shuffle="'every-epoch'", ValidationData=valid_generator,
                CheckpointPath="'{}'".format(self.logdir)
            )))
        else:

            self.engine.evalc("opts = trainingOptions('{}', {});".format(optimizer_name, kwargs_to_string(
                Plots="'training-progress'", Verbose="true", VerboseFrequency=n_stages,
                MaxEpochs=n_epochs, Shuffle="'every-epoch'", CheckpointPath="'{}'".format(self.logdir)
            )))
        self.engine.evalc("net = trainNetwork({}, layers, opts)".format(train_generator))
        self.engine.evalc("save('{}.mat', 'net')".format(self))

    def __call__(self, image):
        """Denoises a batch of images.

        Notes
        -----
        To perform inference on MatlabModels, you need to have a variable on Matlab's workspace called 'net'. This
        variable is the output of a training session, or the result of a load('obj').

        Parameters
        ----------
        image: :class:`numpy.ndarray`
            4D batch of noised images. It has shape: (batch_size, height, width, channels)

        Returns
        -------
        :class:`numpy.ndarray`:
            Restored batch of images, with same shape as the input.

        """
        if image.shape[-1] in [1, 3]:
            # Matlab's convention: height x width x channels x batch_size
            image = np.transpose(image, [1, 2, 3, 0])
        elif image.shape[1] in [1, 3]:
            # Matlab's convention: height x width x channels x batch_size
            image = np.transpose(image, [2, 3, 1, 0])
        # Copy data into Matlab's workspace
        self.engine.workspace['Xbatch'] = matlab.double(image.tolist())
        # Denoises data using net object.
        self.engine.evalc("predicted_image = denoiseImage(Xbatch, net);")
        if not self.return_diff:
            # Returns difference between predicted image and input image.
            self.engine.evalc("predicted_image = Xbatch - predicted_image;")
        return np.transpose(np.array(self.engine.workspace['predicted_image']), [3, 0, 1, 2])

    def __len__(self):
        """Counts the number of parameters in the networks.

        Returns
        -------
        nparams : int
            Number of parameters in the network.
        """
        n_params = 0
        for i in range(1, eng.eval("length(net)") + 1):
            if 'Convolution2DLayer' in eng.eval("class(net.Layers({}))".format(i)):
                n_params += eng.eval("prod(size(net.Layers({}).Weights)) + prod(size(net.Layers({}).Bias))".format(i,
                                                                                                                   i))
            if 'BatchNormalizationLayer' in eng.eval("class(net.Layers({}))".format(i)):
                n_params += eng.eval("prod(size(net.Layers({}).Offset)) + prod(size(net.Layers({}).Scale))".format(i,
                                                                                                                   i))
