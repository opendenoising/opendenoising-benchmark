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


import os
import json
import time
import pandas as pd

from keras import models, optimizers
from OpenDenoising import evaluation
from OpenDenoising.model import module_logger
from OpenDenoising.model import AbstractDeepLearningModel


def history_to_csv(csv_path, history):
    """Saves keras training history to csv

    Args:
        csv_path: 'string': Path to csv file we want to save.
        history: 'History object returned by keras.models.Model.fit_generator

    """
    df = pd.DataFrame(columns=history.history.keys())
    for key in history.history.keys():
        df[key] = history.history[key]
    df.to_csv(os.path.join(csv_path, "training_history.csv"))


class KerasModel(AbstractDeepLearningModel):
    """KerasModel wrapper class.

    Attributes
    ----------
    model : :class:`keras.models.Model`
        Denoiser Keras model used for training and inference.
    return_diff : bool
        If True, return the difference between predicted image, and image at inference time.

    See Also
    --------
    :class:`model.AbstractDenoiser` : for the basic functionalities of Image Denoisers.
    :class:`model.AbstractDeepLearningModel` : for the basic functionalities of Deep Learning based Denoisers.
    """

    def __init__(self, model_name="DeepLearningModel", logdir="./logs/Keras", return_diff=False):
        super().__init__(model_name, logdir, framework="Keras", return_diff=return_diff)

    def charge_model(self, model_function=None, model_path=None, model_weights=None, **kwargs):
        """Keras model charging function.

        There are four main cases for "charge_model" function:

        1. Charge model architecture by using a function "model_function".
        2. Charge model using .json file, previously saved from an existing architecture through the method
           keras.models.Model.to_json().
        3. Charge model using .yaml file, previously saved from an existing architecture through the method
           keras.models.Model.to_yaml()
        4. Charge model using .hdf5 file, previously saved from an existing architecture through the method
           keras.models.Model.save().

        From these four cases, notice that only the last loads the model and the weights at the same time. Therefore,
        at the moment you are loading your model, you should consider specify "model_weights" so that this class can
        find and charge your model weights, which can be saved in both .h5 and .hdf5 formats.

        If this is not the case, and your architecture has not been previously trained, you can run the training and
        then save the weights by using keras.models.Model.save_weights() method, or by using KerasModel.train(),
        method present on this class.

        Parameters
        ----------
        model_function : :class:`function`
            reference to a function that outputs an instance of :class:`keras.models.Model` class.
        model_path : str
            path to model .json, .yaml or .hdf5 file.
        model_weights : str
            path to model .hdf5 weights file.


        Notes
        -----
        If your building function accepts optional arguments, you can specify them by using kwargs.

        Examples
        --------
        Loading Keras DnCNN from class. Notice that in our implementation, depth is an optional argument.

        >>> from OpenDenoising import model
        >>> mymodel = model.KerasModel(model_name="mymodel")
        >>> mymodel.charge_model(model_function=model.architectures.keras.dncnn, depth=17)

        Loading Keras DnCNN from a .hdf5 file.

        >>> from OpenDenoising import model
        >>> mymodel = model.PytorchModel(model_name="mymodel")
        >>> mymodel.charge_model(model_path=PATH)

        Loading Keras DnCNN from a .json + .hdf5 file.

        >>> from OpenDenoising import model
        >>> mymodel = model.PytorchModel(model_name="mymodel")
        >>> mymodel.charge_model(model_path=PATH_TO_JSON, model_weights=PATH_TO_HDF5)
        """
        assert (model_function is not None or model_path is not None), "You should provide at least a model_function\
                                                                        or a model_path to build your neural network\
                                                                        model"
        if model_path is not None:
            filename, extension = os.path.splitext(model_path)
            if extension == ".json" or extension == ".yaml":
                # Opens model architecture file
                with open(model_path, "r") as f:
                    model_file = f.read()
                # Checks the extension (.json or .yaml)
                if extension == ".json":
                    # Loads architecture from .json
                    self.model = models.model_from_json(model_file)
                else:
                    # Loads architecture from .yaml
                    self.model = models.model_from_yaml(model_file)
                # Checks if there are extra weights to load
                if model_weights is not None:
                    self.model.load_weights(model_weights)
                else:
                    module_logger.warning("The model file does not contain any weight information. Be sure to train the"
                                          " network before running your tests.")
            else:
                self.model = models.load_model(model_path)
        else:
            self.model = model_function(**kwargs)
            if model_weights is not None:
                module_logger.info("Loading weights from {}".format(model_weights))
                self.model.load_weights(model_weights)
            else:
                module_logger.warning("You have loaded your model from a python function, which does not hold any "
                                      "information about weight values. Be sure to train the network before running"
                                      " your tests.")
        self.train_info = {
            "TrainTime": -1,
            "Trained": False,
            "NumParams": self.model.count_params()
        }

    def train(self, train_generator, valid_generator=None, n_epochs=1e+2, n_stages=5e+2, learning_rate=1e-3,
              optimizer_name=None, metrics=None, kcallbacks=None, loss=None, valid_steps=10):
        """Function to run the training of a Keras Model.

        Notes
        -----
        There are two cases where training should be launched:

        1. You only loaded your model architecture. In that case, this function will train your model from scratch
           using the dataset specified by train_generator and valid_generator.
        2. You loaded an architecture and weights for your model, but you want to reuse them. It may be the case
           where you want to run your training for a few more epochs, or even perform transfer learning.

        Parameters
        ----------
        train_generator : :class:`data.AbstractDataset`
            Train data generator. Notice that these generators should output paired image samples, the first, a noised
            version of the image, and the second, the ground-truth.
        valid_generator : :class:`data.AbstractDataset`
            Validation data generator
        n_epochs : int
            Number of epochs for which the training will be executed
        n_stages : int
            Number of batches of data are drawn per epoch.
        learning_rate : float
            Initial value for learning rate value for optimization
        optimizer_name : str
            Name of optimizer employed. Check `Keras documentation <https://keras.io/optimizers/>`_
            for more information.
        metrics : list
            List of tensorflow functions implementing scalar metrics (see metrics in evaluation).
        kcallbacks : list
            List of keras callback instances. Consult Keras documentation and evaluation module for more
            information.
        loss : function
            Tensorflow-based loss function. It should take as input two Tensors and output a scalar Tensor holding
            the loss computation.
        valid_steps : int
            Number of batches drawn during evaluation.
        """
        # Default parameters
        if optimizer_name is None:
            optimizer_name = "Adam"
        if loss is None:
            loss = evaluation.tf_mse

        # Retrieve functions from strings
        optimizer_class = getattr(optimizers, optimizer_name)
        optimizer = optimizer_class(lr=learning_rate)
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        # Train the model
        start = time.time()
        self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=n_stages,
            epochs=n_epochs,
            validation_data=valid_generator,
            validation_steps=valid_steps,
            use_multiprocessing=False,
            callbacks=kcallbacks,
            shuffle=True,
            verbose=1
        )
        # Save information to json file
        self.train_info["TrainTime"] = time.time() - start
        self.train_info["Trained"] = True
        with open(os.path.join(self.logdir, self.model_name, "train_info.json"), "w") as f:
            json.dump(self.train_info, f)

    def __call__(self, image):
        """Denoises a batch of images.

        Parameters
        ----------
        image : :class:`numpy.ndarray`
            4D batch of noised images. It has shape: (batch_size, height, width, channels)

        Returns
        -------
        :class:`numpy.ndarray`:
            Restored batch of images, with same shape as the input.

        """
        predicted_image = self.model.predict(image)
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
        return self.model.count_params()
