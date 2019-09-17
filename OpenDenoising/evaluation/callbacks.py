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


import io
import os
import time
import keras
import torch
import shutil
import numpy as np
import tensorflow as tf
import PIL.Image as Image

from keras import backend
from datetime import datetime
from datetime import timedelta
from skimage.util import img_as_ubyte
from OpenDenoising.evaluation import module_logger
from OpenDenoising.data import normalize_batch, clip_batch


def timeit(method):
    """Timing decorator. Use this to get track of time spent on a callback.

    Notes
    -----
    The timing messages have logging.DEBUG level.

    Parameters
    ----------
    method : function
        Method or function to be timed.

    Returns
    -------
    float
        Time spent on function execution
    """
    def timed(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        finish = time.time()

        module_logger.debug("Callback {} took {} seconds".format(method, timedelta(seconds=finish-start)))
        return result
    return timed


class LrSchedulerCallback(keras.callbacks.Callback):
    """Custom Learning Rate scheduler based on Keras. This class is mainly used for compatibility between TfModel,
    PytorchModel and KerasModel. Please note that this class should not be used as a LearningRateScheduler. To specify
    one, you need to use a class that inherits from LrSchedulerCallback.
    """
    def __init__(self):
        super(LrSchedulerCallback, self).__init__()

    @timeit
    def on_epoch_begin(self, epoch, logs=None):
        # Set new learning rate on Keras model
        backend.set_value(self.model.optimizer.lr, self(epoch))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['LearningRate'] = backend.get_value(self.model.optimizer.lr)


class DnCNNSchedule(LrSchedulerCallback):
    """DnCNN learning rate decay scheduler as specified in the original paper.

    After epoch 30, drops the initial learning rate by a factor of 10.
    After epoch 60, drops the initial learning rate by a factor of 20.

    Attributes
    ----------
    initial_lr : float
        Initial learning rate value.
    """
    def __init__(self, initial_lr=1e-3):
        self.initial_lr = initial_lr

    @timeit
    def __call__(self, epoch, logs=None):
        """Calculates the current Learning Rate.

        Parameters
        ----------
        epoch : int
            Current epoch.

        Returns
        -------
        lr : float
            Learning rate value.
        """
        if epoch <= 30:
            lr = self.initial_lr
        elif epoch <= 60:
            lr = self.initial_lr / 10
        elif epoch <= 80:
            lr = self.initial_lr / 20
        else:
            lr = self.initial_lr / 20
        return np.asarray(lr, dtype='float64')


class StepSchedule(LrSchedulerCallback):
    """Drops the learning rate at each 'dropEvery' iterations by a factor of 'factor'.

    Attributes
    ----------
    initial_lr : float
        Initial Learning Rate.
    factor : float
        Decay factor.
    dropEvery : int
        The learning rate will be decayed periodically, where the period is defined by dropEvery.
    """
    def __init__(self, initial_lr=1e-3, factor=0.5, dropEvery=10):
        self.initial_lr = initial_lr
        self.factor = factor
        self.dropEvery = dropEvery

    @timeit
    def __call__(self, epoch, logs=None):
        """Calculates the current Learning Rate.

        Parameters
        ----------
        epoch : int
            Current epoch.

        Returns
        -------
        lr : float
            Learning rate value.
        """
        exp = (epoch + 1) // self.dropEvery

        return np.asarray(self.initial_lr * (self.factor ** exp), dtype="float64")


class PolynomialSchedule(LrSchedulerCallback):
    """Drops the learning rate following a polynomial schedule:

    .. math::
        \\alpha = \\alpha_{0}\\biggr(1 - \dfrac{epoch}{maxEpochs}\\biggr)^{power}

    Attributes
    ----------
    initial_lr : float
        Initial Learning Rate.
    maxEpochs : int
        At the end of maxEpochs, the learning_rate will be zero.
    power : int
        Polynomial power.
    """
    def __init__(self, initial_lr=1e-3, maxEpochs=100, power=1.0):
        self.initial_lr = initial_lr
        self.maxEpochs = maxEpochs
        self.power = power

    @timeit
    def __call__(self, epoch):
        """Calculates the current Learning Rate.

        Parameters
        ----------
        epoch : int
            Current epoch.

        Returns
        -------
        lr : float
            Learning rate value.
        """
        decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
        lr = self.initial_lr * decay

        return np.asarray(lr, dtype="float64")


class ExponentialSchedule(LrSchedulerCallback):
    """Drops the learning rate following a exponential schedule:

    .. math::
        \\alpha = \\alpha_{0}\\times\\gamma^{epoch}

    Attributes
    ----------
    initial_lr : float
        Initial Learning Rate.
    factor : float
        Rate at which the learning rate is decayed at each epoch.
    """
    def __init__(self, initial_lr=1e-3, gamma=0.5):
        self.initial_lr = initial_lr
        self.gamma = gamma

    @timeit
    def __call__(self, epoch):
        """Calculates the current Learning Rate.

        Parameters
        ----------
        epoch : int
            Current epoch.

        Returns
        -------
        lr : float
            Learning rate value.
        """
        lr = self.initial_lr * (gamma ** epoch)
        lr = np.asarray(lr, dtype="float64")

        return lr


def make_image(image_tensor):
    """Convert image to Tensorboard image summary.

    Parameters
    ----------
    image_tensor : :class:`numpy.ndarray`
        Image to be shown in Tensorboard.

    Returns
    -------
    :tf.Tensor:
        Tensor of type string, containing the serialized image Summary protocol buffer.
    """
    h, w, c = image_tensor.shape
    image = Image.fromarray(image_tensor.reshape(h, w))
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()

    return tf.Summary.Image(height=h, width=w, colorspace=c,
                            encoded_image_string=image_string)


def stack_images(list_of_images):
    """Stacks noisy + clean + denoised images in the horizontal, then stacks the entire batch in the vertical

    Parameters
    ----------
    list_of_images : list
        List of :class:`numpy.ndarray` objects.

    Returns
    -------
    stacked : :class:`numpy.ndarray`
        Numpy array corresponding to the stacked images.
    """
    to_stack = []
    # Stacks batch on the vertical
    for image in list_of_images:
        to_stack.append(np.vstack([image[i] for i in range(len(image))]))
    # Stacks images on the horizontal
    stacked = np.hstack(to_stack)

    return stacked


class TensorboardImage(keras.callbacks.Callback):
    """Tensorboard Keras callback. At each epoch end, it plots on Tensorboard metrics/loss information on validation
    data, as well as a denoising summary.

    Attributes
    ----------
    valid_generator : :class:`data.AbstractDatasetGenerator`
        Image dataset generator. Provide validation data for model evaluation.
    denoiser : :class:`model.AbstractDeepLearningModel`
        Image denoising object.
    folder_string : str
        String containing the path to logging directory. Corresponds to denoiser's name.
    """
    def __init__(self, valid_generator, denoiser, preprocess="clip"):
        super().__init__()
        # Folder string for tensorboard files
        self.folder_string = str(denoiser)
        # Checks if folder already exists
        if not os.path.isdir(os.path.join(denoiser.logdir, self.folder_string)):
            os.makedirs(os.path.join(denoiser.logdir, self.folder_string))
        # Tensorboard objects
        self.writer = tf.summary.FileWriter(
            os.path.join(denoiser.logdir, self.folder_string)
        )
        self.writer.add_graph(keras.backend.get_session().graph)
        # Auxiliary objects
        self.seen = 0  # Number of seen epochs
        self.denoiser = denoiser  # Denoiser model
        self.valid_generator = valid_generator
        self.preprocess = preprocess

    @timeit
    def on_epoch_end(self, epoch, logs=None):
        """Method called at each epoch end. Evaluates the denoiser at validation data, then plots metrics/loss into
        tensorboard. Shows denoising visual results on Tensorboard.

        Parameters
        ----------
        epoch : int
            Current evaluation epoch.
        logs : dict
            Dictionary where the keys are metrics/loss names, and the values are the functions to be evaluated.
        """
        self.seen += 1  # Updates intern epoch counter

        if logs is not None:
            # Tensorboard summaries:
            for key in logs:
                summary = tf.Summary(
                    value=[tf.Summary.Value(tag=key, simple_value=logs[key])]
                )
                self.writer.add_summary(summary, epoch)

        i = np.random.randint(0, len(self.valid_generator))
        img_noise, img_ref = self.valid_generator[i]
        img_denoised = self.denoiser(img_noise)

        # Converts 8 first samples to uint8
        if self.preprocess == "normalize":
            noised = img_as_ubyte(normalize_batch(img_noise))
            original = img_as_ubyte(normalize_batch(img_ref))
            denoised = img_as_ubyte(normalize_batch(img_denoised))
        elif self.preprocess == "clip":
            noised = img_as_ubyte(clip_batch(img_noise))
            original = img_as_ubyte(clip_batch(img_ref))
            denoised = img_as_ubyte(clip_batch(img_denoised))
        else:
            noised = img_as_ubyte(img_noise)
            original = img_as_ubyte(img_ref)
            denoised = img_as_ubyte(img_denoised)

        # Print to tensorboard
        img = make_image(stack_images([noised, original, denoised]))
        summary_original = tf.Summary(
            value=[tf.Summary.Value(tag="Denoising Result", image=img)]
        )
        self.writer.add_summary(summary_original, epoch)


def tf_saved_model(directory, session, input, is_training, output, epoch):
    if len(os.listdir(directory)) > 0:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                os.unlink(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)
    tf.saved_model.simple_save(session=session, export_dir=directory,
                               inputs={input.name.replace(":0", ""): input,
                                       is_training.name.replace(":0", ""): is_training},
                               outputs={output.name.replace(":0", ""): output})


class CheckpointCallback(keras.callbacks.Callback):
    """Creates training checkpoints for Deep Learning models.

    Attributes
    ----------
    denoiser : :class:`model.AbstractDenoiser`
        Denoiser object to be saved.
    monitor : str
        Name of the metric being tracked. If the name is not present on logs, tracks the loss value.
    mode : str
        String having one of these two values: {'max', 'min'}. If it is 'max', saves the model with the greater metric
        value. If it is 'min', saves the model with the smaller metric value.
    period : int
        Saves models at uniform intervals specified by period.
    logdir : str
        String containing the path to the logs directory.
    """
    def __init__(self, denoiser, monitor="loss", mode='max', period=1):
        logpath = os.path.join(denoiser.logdir, str(denoiser))
        try:
            os.makedirs(logpath)
        except FileExistsError:
            module_logger.warning("Directory {} already exists. Nothing was done.".format(logpath))
        try:
            os.makedirs(os.path.join(logpath, "ModelFiles"))
        except FileExistsError:
            module_logger.warning("Directory {} already exists. Nothing was done.".format(os.path.join(logpath,
                                                                                                       "ModelFiles")))

        self.denoiser = denoiser
        self.monitor = monitor
        self.mode = mode
        self.period = period
        self.logpath = logpath
        self.best_value = np.inf if mode == 'min' else -np.inf

    @timeit
    def on_epoch_end(self, epoch, logs):
        """Read logs, determine if the model should be saved based on period, best_value and mode."""
        if epoch % self.period == 0:
            try:
                metric_val = logs[self.monitor]
            except KeyError:
                module_logger.warning("Monitor {} not present in dictionary logs with keys {}." \
                                      " Falling into loss tracking.".format(self.monitor, list(logs.keys())))
                self.mode = 'min'
                self.best_value = np.inf
                self.monitor = 'loss'
                metric_val = logs[self.monitor]
            if metric_val > self.best_value and self.mode == "max":
                self.best_value = metric_val
                self.__save_model(epoch)
            elif metric_val < self.best_value and self.mode == "min":
                self.best_value = metric_val
                self.__save_model(epoch)

    def __save_model(self, epoch):
        """Saves the denoiser model based on its class. """
        model_class = self.denoiser.__class__.__name__
        if model_class == "TfModel":
            tf_saved_model(os.path.join(self.logpath, "ModelFiles"),
                           self.denoiser.tf_session,
                           self.denoiser.model_input,
                           self.denoiser.is_training,
                           self.denoiser.model_output,
                           epoch)
        elif model_class == "KerasModel":
            model_json = self.denoiser.model.to_json()
            # Save model architecture in a JSON file
            with open(os.path.join(self.logpath, "ModelFiles", str(self.denoiser) + ".json"), "w") as f:
                f.write(model_json)
            # Save model weights on .HDF5 file
            self.denoiser.model.save_weights(
                os.path.join(self.logpath, "ModelFiles", str(self.denoiser) + "_weights.hdf5")
            )
        elif model_class == "PytorchModel":
            torch.save(self.denoiser.model, os.path.join(self.logpath, "ModelFiles", str(self.denoiser) + ".pth"))
        else:
            raise ValueError("Expected model to be either PytorchModel, TfModel or KerasModel, but got"\
                             "{}".format(model_class))
