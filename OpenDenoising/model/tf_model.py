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
import json
import time
import shutil
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from datetime import timedelta
from OpenDenoising import evaluation
from OpenDenoising.model import module_logger
from OpenDenoising.model import AbstractDeepLearningModel

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def write_to_csv(csv_path, row):
    row_str = ''
    for element in row:
        row_str += '{}, '.format(element)
    row_str += '\n'
    with open(os.path.join(csv_path, "log.csv"), "a") as csvFile:
        csvFile.write(row_str)


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


def tf_saved_model(directory, session, input, is_training, output, epoch):
    persistence_dirname = "_persistence"
    if "_persistence" not in os.listdir(directory):
        os.mkdir(os.path.join(directory, persistence_dirname))
    elif len(os.listdir(os.path.join(directory, persistence_dirname))) > 0:
        for filename in os.listdir(os.path.join(directory, persistence_dirname)):
            filepath = os.path.join(directory, persistence_dirname, filename)
            if os.path.isfile(filepath):
                os.unlink(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)
    tf.saved_model.simple_save(session=session,
                               export_dir=os.path.join(directory, persistence_dirname),
                               inputs={input.name.replace(":0", ""): input,
                                       is_training.name.replace(":0", ""): is_training},
                               outputs={output.name.replace(":0", ""): output})


class TfModel(AbstractDeepLearningModel):
    """Tensorflow model class wrapper.

    Parameters
    ----------
    loss : :class:`tf.Tensor`
        Tensor holding the computation for the loss function.
    saver : :class:`tf.train.Saver`
        Object for saving the model at each epoch iteration.
    metrics : list
        List of :class:`tf.Tensor` objects holding the computation for each metric.
    opt_step : :class:`tf.Operation`
        Tensorflow operation corresponding to the update performed on model's variables.
    tf_session : :class:`tf.Session`
        Instance of tensorflow session.
    model_input : :class:`tf.Tensor`
        Tensor corresponding to model's input.
    is_training : :class:`tf.Tensor`
        If batch normalization is used, corresponds to a tf placeholder controlling training and inference phases of
        batch normalization.
    model_output : :class:`tf.Tensor`
        Tensor corresponding to model's output.
    ground_truth : :class:`tf.placeholder`
        Placeholder corresponding to original training images (clean).

    See Also
    --------
    :class:`model.AbstractDenoiser` : for the basic functionalities of Image Denoisers.
    :class:`model.AbstractDeepLearningModel` : for the basic functionalities of Deep Learning based Denoisers.
    """
    def __init__(self, model_name="DeepLearningModel", logdir="./logs/Tensorflow", return_diff=False):
        super().__init__(model_name, logdir, framework="Tensorflow", return_diff=return_diff)

        # Auxiliary tensorflow variables
        self.loss = None
        self.saver = None
        self.metrics = None
        self.opt_step = None
        self.tf_session = None
        self.model_input = None
        self.is_training = None
        self.model_output = None
        self.ground_truth = None

    def charge_model(self, model_function=None, model_path=None, **kwargs):
        """Charges Tensorflow model into the wrapper class by using a model file, or a building architecture function.

        Parameters
        ----------
        model_function : :class:`function`
            Building architecture function, which returns at least two tensors: one for the graph input, and another for
            the graph output.
        model_path : str
            String containing the path to a .pb or .meta file, holding the computational graph for the model. Note that
            each of these files correspond to a different tensorflow saving API.

            * .pb files correspond to saved_model API. This API saves the model in a folder containing the .pb file,
              along with a folder called "variables", holding the weight values.
            * .meta files correspond to tf.train API. This API saves the model through four files (.meta, .index, .data
              and checkpoint).

            You can save your models in one of these two formats.

        Notes
        -----
        This function accepts Keyword arguments which can be used to pass additional parameters to model_function.

        Examples
        --------
        Loading Tensorflow DnCNN from class. Notice that in our implementation, depth is an optional argument.

        >>> from OpenDenoising import model
        >>> mymodel = model.TfModel(model_name="mymodel")
        >>> mymodel.charge_model(model_function=model.architectures.tensorflow.DnCNN, depth=17)

        Loading Tensorflow DnCNN from a file. Note that the file which you are going to charge on TfModel need to be a
        .pb or a .meta file. In the first case, we assume you have used the `SavedModel API
        <https://www.tensorflow.org/guide/saved_model>`_, while in the second case, we assume the `Checkpoint API
        <https://www.tensorflow.org/guide/checkpoints>`_.

        >>> from OpenDenoising import model
        >>> mymodel = model.PytorchModel(model_name="mymodel")
        >>> mymodel.charge_model(model_path=PATH_TO_PB_OR_META)
        """
        assert (model_function is not None or model_path is not None), "You should provide at least a model_function\
                                                                        or a model_path to build your neural network\
                                                                        model"
        if model_function:
            self.tf_session = tf.Session(config=config)
            model_function(**kwargs)
            module_logger.warning("You have loaded your model from a python function, which does not hold any "
                                 "information about weight values. Be sure to train the network before running"
                                 " your tests.")
        if model_path:
            _, extension = os.path.splitext(model_path)
            if extension == ".pb":
                print("Loading model using SavedModel API.")
                # saved_model API
                module_logger.info("Loading model from tensorflow ProtoBuf")
                self.tf_session = tf.Session(config=config)
                tf.saved_model.load(self.tf_session, ["serve"], os.path.dirname(model_path))
            elif extension == '.meta':
                print("Loading model using Checkpoint API.")
                # tf.train API.
                module_logger.info("Loading model from tensorflow checkpoint")
                self.tf_session = tf.Session(config=config)
                self.saver = tf.train.import_meta_graph(model_path)
                self.saver.restore(self.tf_session, tf.train.latest_checkpoint(os.path.dirname(model_path)))

        nodes = [n for n in tf.get_default_graph().as_graph_def().node]
        try:
            input_name = [n.name for n in nodes if "input" in n.name.lower()][0]
        except IndexError as err:
            raise ValueError("Could not find an input in the computational graph. Make sure your model has "
                             "'input' in his input Tensor")

        try:
            is_training_name = [n.name for n in nodes if "is_training" in n.name.lower()][0]
        except IndexError as err:
            raise ValueError("Could not find a Tensor for determining the training phase in Batch Normalization"
                             "layer")

        try:
            output_name = [n.name for n in nodes if "output" in n.name.lower()][0]
        except IndexError as err:
            raise ValueError("Could not find an output in the computational graph. Make sure your model has 'output' in"
                             " his output Tensor")

        # Get tensors by name
        self.model_input = tf.get_default_graph().get_tensor_by_name("{}:0".format(input_name))
        self.is_training = tf.get_default_graph().get_tensor_by_name("{}:0".format(is_training_name))
        self.model_output = tf.get_default_graph().get_tensor_by_name("{}:0".format(output_name))

        self.ground_truth = tf.placeholder(tf.float32, self.model_output.get_shape().as_list())
        self.train_info = {
            "TrainTime": -1.,
            "Trained": False,
            "NumParams": len(self)
        }

    def train(self, train_generator, valid_generator=None, n_epochs=250, n_stages=500, learning_rate=1e-3,
              metrics=None, optimizer_name=None, kcallbacks=None, loss=None, valid_steps=10,
              saving_api="SavedModel"):
        """Trains a tensorflow model.

        Parameters
        ----------
        train_generator : data.AbstractDataset
            Dataset object inheriting from AbstractDataset class. It is a generator object that yields
            data from train dataset folder.
        valid_generator : data.AbstractDataset
            Dataset object inheriting from AbstractDataset class. It is a generator object that yields
            data from valid dataset folder.
        n_epochs : int
            Number of training epochs.
        n_stages : int
            Number of batches seen per epoch.
        learning_rate : float
            Constant multiplication constant for optimizer or initial value for training with
            dynamic learning rate (see callbacks)
        metrics : list
            List of tensorflow functions implementing scalar metrics (see metrics in evaluation).
        optimizer_name : str
            Name of optimizer to use. Check tf.train documentation for a complete list.
        kcallbacks : list
            List of callbacks.
        loss : :class:`function`
            Tensorflow-based loss function. It should take as input two Tensors and output a scalar Tensor holding
            the loss computation.
        valid_steps : int
            If valid_generator was specified, valid_steps determines the number of valid batches
            that will be seen per validation run.
        saving_api : string
            If training_api = "tftrain", uses tf.train.Saver as the model saver. Otherwise, uses saved_model API.
        """
        do_valid = bool(valid_generator)

        """ Optimizer """
        if optimizer_name is None:
            # Checks if optimizer was given
            optimizer_name = "AdamOptimizer"

        """ Loss """
        if loss is None:
            self.loss = evaluation.tf_se(self.ground_truth, self.model_output)
        else:
            self.loss = loss(self.ground_truth, self.model_output)

        """ Metrics """
        self.metrics = [metric(self.ground_truth, self.model_output) for metric in metrics]

        # callbacks
        if kcallbacks is None:
            # Checks if callbacks were given
            kcallbacks = []

        # Adding optimization variables to the graph
        lr = tf.placeholder(tf.float32, [])
        optimizer_obj = getattr(tf.train, optimizer_name)
        optimizer = optimizer_obj(learning_rate=lr)

        # Use update_ops with control_dependencies is necessary for batch normalization to work.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt_step = optimizer.minimize(self.loss)

        # Initializes training operation
        if self.saver is None and saving_api == "tftrain":
            self.saver = tf.train.Saver(name="model")

        # Initializes global variables
        self.tf_session.run(tf.global_variables_initializer())

        # Initializes memory of best_epochs
        min_val_loss = np.inf
        best_epoch = None

        start = time.time()
        write_to_csv(os.path.join(self.logdir, str(self)), ['Epoch', 'Loss'] + [metric.__name__ for metric in metrics] +
                     ["val_" + metric.__name__ for metric in metrics])
        log_dict = {"n_epochs": n_epochs}
        for i in range(int(n_epochs)):
            pgbar = tqdm(range(n_stages), ncols=150, ascii=True)
            epoch_start = time.time()
            log_dict["start_time"] = epoch_start
            log_dict["epoch"] = i
            log_dict["best_epoch"] = best_epoch
            logs = dict()
            logs["LearningRate"] = learning_rate

            """ Training loop """
            for _ in pgbar:
                input_batch, output_batch = next(train_generator)
                _, loss_v, metrics_v = self.tf_session.run([self.opt_step, self.loss, self.metrics],
                                                           feed_dict={self.model_input: input_batch,
                                                                      self.is_training: True,
                                                                      self.ground_truth: output_batch,
                                                                      lr: learning_rate})
                log_dict["loss_val"] = loss_v
                log_dict["metrics_val"] = metrics_v
                log_dict["finish_time"] = time.time()
                change_pgbar_desc(pgbar, log_dict)
            logs["loss"] = loss_v
            for metric_v, metric in zip(metrics_v, metrics):
                logs[metric.__name__] = metric_v

            """ Evaluation loop """
            if do_valid:
                loss_m = 0
                metrics_m = np.zeros((len(metrics)))
                pgbar = tqdm(range(valid_steps), ncols=150, ascii=True)
                eval_start = time.time()
                log_dict["start_time"] = epoch_start
                for _ in pgbar:
                    input_batch, output_batch = next(valid_generator)
                    loss_v, metrics_v = self.tf_session.run([self.loss, self.metrics],
                                                            feed_dict={self.model_input: input_batch,
                                                                       self.is_training: False,
                                                                       self.ground_truth: output_batch})
                    log_dict["loss_val"] = loss_v
                    log_dict["metrics_val"] = metrics_v
                    log_dict["finish_time"] = time.time()
                    change_pgbar_desc(pgbar, log_dict, train=False)
                    loss_m = loss_m + loss_v / valid_steps
                    metrics_m = metrics_m + np.array(metrics_v) / valid_steps

                logs["val_loss"] = loss_m
                for metric_m, metric in zip(metrics_m, metrics):
                    logs["val_" + metric.__name__] = metric_m
            else:
                loss_m = loss_v

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
                else:
                    # Calling other callbacks (such as Tensorboard).
                    callback.on_epoch_end(i, logs=logs)

        self.train_info["TrainTime"] = time.time() - start
        self.train_info["Trained"] = True
        with open(os.path.join(self.logdir, self.model_name, "train_info.json"), "w") as f:
            json.dump(self.train_info, f)

    def __call__(self, image):
        """Denoises a batch of images.

        Parameters
        ----------
        image: :class:`np.ndarray`
            4D batch of noised images. It has shape: (batch_size, height, width, channels)

        Returns
        -------
        :class:`np.ndarray`:
            Restored batch of images, with same shape as the input.

        """
        predicted_image = self.tf_session.run(self.model_output, feed_dict={self.model_input: image,
                                                                            self.is_training: False})
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
        n_params = 0
        for var in tf.trainable_variables():
            shape = var.get_shape().as_list()
            var_params = 1
            for dim in shape:
                var_params = var_params * dim
            n_params = n_params + var_params
        return n_params
