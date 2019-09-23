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
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from datetime import timedelta
from OpenDenoising.model import module_logger
from OpenDenoising.model import AbstractDeepLearningModel


def change_pgbar_desc(pgbar, log, train=True):
    if train:
        string = "[Train {}/{}, Best: {}] ".format(log["epoch"], log["n_epochs"], log["best_epoch"])
        string += "Loss: {0:.4f}, Metrics: ".format(log["loss_val"])
        for metric in log["metrics_val"]:
            string += "{0:.4f} ".format(metric)
        string += " Elapsed Time: {}".format(timedelta(seconds=log["finish_time"] - log["start_time"]))
    else:
        string = "[Valid {}/{}] ".format(log["epoch"], log["n_epochs"])
        string += "Loss: {0:.4f}, Metrics: ".format(log["loss_val"])
        for metric in log["metrics_val"]:
            string += "{0:.4f} ".format(metric)
        string += " Elapsed Time: {}".format(timedelta(seconds=log["finish_time"] - log["start_time"]))
    pgbar.set_description(string)


class PytorchModel(AbstractDeepLearningModel):
    """Pytorch wrapper class.

    See Also
    --------
    :class:`model.AbstractDenoiser` : for the basic functionalities of Image Denoisers.
    :class:`model.AbstractDeepLearningModel` : for the basic functionalities of Deep Learning based Denoisers.
    """
    def __init__(self, model_name="DeepLearningModel", logdir="./logs/Pytorch", return_diff=False):
        super().__init__(model_name, logdir, framework="Pytorch", return_diff=return_diff)

    def charge_model(self, model_function=None, model_path=None, **kwargs):
        """Pytorch model charging function. You can charge a model either by specifying a class that implements the
        network architecture (passing it through model_function) or by specifying the path to a .pt or .pth file.
        If you class constructor accepts optional arguments, you can specify these by using Keyword arguments.

        Parameters
        ----------
        model_function : :class:`torch.nn.Module`
            Pytorch network Class implementing the network architecture.
        model_path : str
            String containing the path to a .pt or .pth file.

        Examples
        --------

        Loading Pytorch DnCNN from class. Notice that in our implementation, depth is an optional argument.

        >>> from OpenDenoising import model
        >>> mymodel = model.PytorchModel(model_name="mymodel")
        >>> mymodel.charge_model(model_function=model.architectures.pytorch.DnCNN, depth=17)

        Loading Pytorch DnCNN from a file.

        >>> from OpenDenoising import model
        >>> mymodel = model.PytorchModel(model_name="mymodel")
        >>> mymodel.charge_model(model_path=PATH)
        """
        assert (model_function is not None or model_path is not None), "You should provide at least a model_function\
                                                                        or a model_path to build your neural network\
                                                                        model"
        if model_path is not None:
            filename, extension = os.path.splitext(model_path)
            if extension == '.pt' or extension == '.pth':
                module_logger.info("Loading Pytorch model from {}".format(model_path))
                self.model = torch.load(model_path)
            else:
                raise ValueError("Invalid file extension. Expected .pt or .pth, but got {}".format(extension))

        elif model_function is not None:
            self.model = model_function(**kwargs)
            module_logger.warning("You have loaded your model from a python function, which does not hold any "
                                  "information about weight values. Be sure to train the network before running"
                                  " your tests.")
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def train(self, train_generator, valid_generator=None, n_epochs=250, n_stages=500, learning_rate=1e-3,
              metrics=None, optimizer_name=None, kcallbacks=None, loss=None, valid_steps=10,):
        """Trains a Pytorch model.

        Parameters
        ----------
        train_generator : data.AbstractDataset
            dataset object inheriting from AbstractDataset class. It is a generator object that yields
            data from train dataset folder.
        valid_generator : data.AbstractDataset
            dataset object inheriting from AbstractDataset class. It is a generator object that yields
            data from valid dataset folder.
        n_epochs : int
            number of training epochs.
        n_stages : int
            number of batches seen per epoch.
        learning_rate : float
            constant multiplication constant for optimizer or initial value for training with
            dynamic learning rate (see callbacks)
        metrics : list
            List of metric functions. These functions should have two inputs, two instances of :class:`numpy.ndarray`.
            It outputs a float corresponding to the metric computed on those two arrays. For more information, take a
            look on the Benchmarking module.
        optimizer_name : str
            Name of optimizer to use. Check Pytorch documentation for a complete list.
        kcallbacks : list
            List of custom_callbacks.
        loss : :class:`torch.nn.modules.loss`
            Pytorch loss function.
        valid_steps : int
            If valid_generator was specified, valid_steps determines the number of valid batches
            that will be seen per validation run.
        """
        do_valid = bool(valid_generator)

        """ Optimizer """
        if optimizer_name is None:
            # Checks if optimizer was given
            optimizer_name = "Adam"
        optimizer = getattr(optim, optimizer_name)(self.model.parameters(), lr=learning_rate)

        """ Callbacks """
        if kcallbacks is None:
            kcallbacks = []

        """ Loss """
        if loss is None:
            loss = nn.MSELoss(reduction="mean")

        min_val_loss = np.inf
        best_epoch = None

        log_dict = {"n_epochs": n_epochs}
        for i in range(int(n_epochs)):
            pgbar = tqdm(range(n_stages), ncols=150, ascii=True)
            epoch_start = time.time()
            log_dict["start_time"] = epoch_start
            log_dict["epoch"] = i
            log_dict["best_epoch"] = best_epoch
            callback_logs = dict()
            callback_logs["LearningRate"] = learning_rate

            for _ in pgbar:
                # Sets model to training
                self.model.train()
                # Resets gradients
                self.model.zero_grad()
                optimizer.zero_grad()

                # Get image batches
                x_numpy, y_numpy = next(train_generator)
                if x_numpy.shape[1] not in [1, 3]:
                    x_numpy = np.transpose(x_numpy, [0, 3, 1, 2])
                if y_numpy.shape[1] not in [1, 3]:
                    y_numpy = np.transpose(y_numpy, [0, 3, 1, 2])
                # Numpy array to Tensor
                x_tensor = torch.from_numpy(x_numpy).float()
                y_tensor = torch.from_numpy(y_numpy).float()
                if torch.cuda.is_available():
                    # Pass Tensors to GPU
                    x_tensor = x_tensor.cuda()
                    y_tensor = y_tensor.cuda()
                # Makes prediction based on inputs
                y_pred_tensor = self.model(x_tensor)
                # Compute loss
                loss_tensor = loss(y_pred_tensor, y_tensor)
                # Tensor to scalar
                log_dict["loss_val"] = loss_tensor.item()
                # Computes metrics
                if torch.cuda.is_available():
                    # Pass prediction to CPU
                    y_pred_tensor = y_pred_tensor.cpu()
                y_pred_numpy = y_pred_tensor.detach().numpy()
                metrics_v = []
                for metric in metrics:
                    metrics_v.append(metric(y_numpy, y_pred_numpy))
                log_dict["metrics_val"] = metrics_v
                log_dict["finish_time"] = time.time()
                change_pgbar_desc(pgbar, log_dict)
                
                # Computes gradients
                loss_tensor.backward()
                # Performs optimization
                optimizer.step()

            callback_logs['loss'] = loss_tensor.item()
            for metric_v, metric in zip(metrics_v, metrics):
                callback_logs[metric.__name__] = metric_v

            if do_valid:
                self.model.eval()
                loss_m = 0
                metrics_m = np.zeros((len(metrics)))
                pgbar = tqdm(range(valid_steps), ncols=150, ascii=True)
                eval_start = time.time()
                log_dict["start_time"] = eval_start
                with torch.no_grad():
                    for _ in pgbar:
                        x_numpy, y_numpy = next(valid_generator)
                        if x_numpy.shape[1] not in [1, 3]:
                            x_numpy = np.transpose(x_numpy, [0, 3, 1, 2])
                        if y_numpy.shape[1] not in [1, 3]:
                            y_numpy = np.transpose(y_numpy, [0, 3, 1, 2])
                        # Numpy array to Tensor
                        x_tensor = torch.from_numpy(x_numpy).float()
                        y_tensor = torch.from_numpy(y_numpy).float()
                        if torch.cuda.is_available():
                            # Pass prediction to CPU
                            x_tensor = x_tensor.cuda()
                            y_tensor = y_tensor.cuda()
                        # Makes prediction based on inputs
                        y_pred_tensor = self.model(x_tensor)
                        # Compute loss
                        loss_tensor = loss(y_pred_tensor, y_tensor)
                        # Tensor to scalar
                        log_dict["loss_val"] = loss_tensor.item()
                        # Computes metrics
                        if torch.cuda.is_available():
                            # Pass prediction to CPU
                            y_pred_tensor = y_pred_tensor.cpu()
                        y_pred_numpy = y_pred_tensor.detach().numpy()
                        metrics_v = []
                        for metric in metrics:
                            metrics_v.append(metric(y_numpy, y_pred_numpy))
                        log_dict["metrics_val"] = metrics_v
                        log_dict["finish_time"] = time.time()
                        change_pgbar_desc(pgbar, log_dict, train=False)
                        loss_m = loss_m + loss_tensor.item() / valid_steps
                        metrics_m = metrics_m + np.array(metrics_v) / valid_steps
                    callback_logs['val_loss'] = loss_m
                    for metric_m, metric in zip(metrics_m, metrics):
                        callback_logs["val_" + metric.__name__] = metric_m
            else:
                loss_m = loss_tensor.item()

            if loss_m < min_val_loss:
                min_val_loss = loss_m
                best_epoch = i

            """ Calls on_epoch_end on callbacks and datasets. """
            train_generator.on_epoch_end()
            if valid_generator is not None:
                valid_generator.on_epoch_end()
            for callback in kcallbacks:
                if "schedule" in callback.__class__.__name__.lower():
                    # Calling learning rate scheduler.
                    learning_rate = callback(i)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate
                else:
                    # Calling other callbacks (such as Tensorboard).
                    callback.on_epoch_end(i, logs=callback_logs)

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
        self.model.eval()
        channels_first = True
        if image.shape[1] not in [1, 3]:
            # Pytorch only accepts NCHW input arrays.
            # If dim1 is not 1 or 3 (channel), transposes the array.
            channels_first = False
            image = np.transpose(image, [0, 3, 1, 2])
        image_tensor = torch.from_numpy(image).float()
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        denoised_tensor = self.model(image_tensor)
        if torch.cuda.is_available():
            denoised_tensor = denoised_tensor.cpu()
        denoised_numpy = denoised_tensor.detach().numpy()
        if not channels_first:
            # Transforms to NHWC.
            # Note: output dimension should agree with input dimension.
            denoised_numpy = np.transpose(denoised_numpy, [0, 2, 3, 1])
        if self.return_diff:
            return image - denoised_numpy
        else:
            return denoised_numpy

    def __len__(self):
        """Counts the number of parameters in the network.

        Returns
        -------
        nparams : int
            Number of parameters in the network.
        """
        n_params = 0
        for param in self.model.parameters():
            var_params = 1
            for dim in list(param.shape):
                var_params *= dim
            n_params += var_params
        return n_params
