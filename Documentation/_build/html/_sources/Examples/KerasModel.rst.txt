Keras Model tutorial
====================

This tutorial is a part of Model module guide. Here, we explore how you
can use the FilteringModel wrapper to use your Python or Matlab
filtering functions in the benchmark.

First, being on the project's root, you need to import the necessary modules,

.. code-block:: python

    import gc
    import keras
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt

    from functools import partial
    from OpenDenoising import data
    from OpenDenoising import model
    from keras import layers, models
    from OpenDenoising import evaluation

    eng = matlab.engine.start_matlab()

The following function will be used throughout this tutorial to display denoising results,

.. code-block:: python

    def display_results(clean_imgs, noisy_imgs, rest_images, name):
        """Display denoising results."""
        fig, axes = plt.subplots(5, 3, figsize=(15, 15))

        plt.suptitle("Denoising results using {}".format(name))

        for i in range(5):
            axes[i, 0].imshow(np.squeeze(clean_imgs[i]), cmap="gray")
            axes[i, 0].axis("off")
            axes[i, 0].set_title("Ground-Truth")

            axes[i, 1].imshow(np.squeeze(noisy_imgs[i]), cmap="gray")
            axes[i, 1].axis("off")
            axes[i, 1].set_title("Noised Image")

            axes[i, 2].imshow(np.squeeze(rest_imgs[i]), cmap="gray")
            axes[i, 2].axis("off")
            axes[i, 2].set_title("Restored Images")

Moreover, you may download the data we will use by using the following function,

.. code-block:: python

    data.download_BSDS_grayscale(output_dir="./tmp/BSDS500/")

The models will be evaluated using the BSDS dataset,

.. code-block::

    # Training images generator
    train_generator = data.DatasetFactory.create(path="./tmp/BSDS500/Train",
                                                 batch_size=8,
                                                 n_channels=1,
                                                 noise_config={data.utils.gaussian_noise: [25]},
                                                 preprocessing=[partial(data.gen_patches, patch_size=40),
                                                                partial(data.dncnn_augmentation, aug_times=1)],
                                                 name="BSDS_Train")


.. code-block:: python

    # Validation images generator
    valid_generator = data.DatasetFactory.create(path="./tmp/BSDS500/Valid",
                                                 batch_size=8,
                                                 n_channels=1,
                                                 noise_config={data.utils.gaussian_noise: [25]},
                                                 name="BSDS_Valid")

To execute multiple models that access the GPU, you need to allow Tensorflow/Keras to allocate memory only when
needed. This is done through,

.. code:: ipython3

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.set_session(session)

Keras offers two ways to construct a model, whether by explicitly
programming it using their API, or by using a file to load the
computational graph. Once you have constructed your model, you will have
at hand a “keras.models.Model” class instance.

Since the use of different frameworks is not the same, to force Keras
models to adequate their functionality to the Benchmark needs, we
provide the KerasModel class, that redefines some of the Keras API
functionalities.

Charging a model
----------------

The first step to build a KerasModel instance, is to effectively charge
a keras model (“keras.models.Model”) into the class. This is done
through the method “**charge_model**”. There are two ways to charge the
model into the wrapper class: by using a function, or by using a file.
These two cases are managed by the use of three parameters of the method
“**charge_model**”:

-  **model_function**: This argument receives a function object (with
   \__call_\_ method defined). The function object is responsable to
   build the Keras model inside the class.
-  **model_path**: This argument is a string containing the path to a
   .hdf5 file (weights + architecture) or a .json/.yaml file
   (architecture).
-  **model_weights**: If you passed the model architecture through a
   .json file, and you do have a .hdf5 containing weights only, you can
   pass the path to the .hdf5 weight file using the “model_weights”
   parameter.

From a function
---------------

To charge a “keras.models.Model” into the wrapper class, you need to
explicitly program the Keras model. To do so, you should provide to the
method “**charge_model**” a function that returns an instance of
“keras.models.Model” class corresponding to your architecture. As an
example, consider the following implementation of `DnCNN
network <https://arxiv.org/pdf/1608.03981.pdf>`__:

.. code:: python

   def dncnn():
       x = layers.InputLayer(shape=[None, None, 1])
       y = layers.Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same')(x)
       y = layers.Activation("relu")(y)

       # Middle layers: Conv + ReLU + BN
       for i in range(1, 16):
           y = layers.Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same', use_bias=False)(y)
           y = layers.BatchNormalization(axis=-1, momentum=0.0, epsilon=1e-3)(y)
           y = layers.Activation("relu")(y)

       y = layers.Conv2D(filters=1, kernel_size=5, strides=(1, 1), use_bias=False, padding='same')(y)
       y = layers.Subtract()([x, y])

       # Keras model
       return models.Model(x, y)

additionally to this example, you should consider the following
convention to architecture functions:

.. code:: python

   def my_arch_func(optional arguments):
       # Steps to build your Keras model
       return keras.models.Model(inputs, outputs)

In the following blocks of code, we show how we can charge the model
into a “KerasModel” wrapper class by using the “dncnn” function.

.. code:: ipython3

    # Creating the KerasModel instance
    kerasmodel_ex1 = model.KerasModel(model_name="Example1", logdir="../../logs/Keras")
    print("KerasModel {} created succesfully.".format(kerasmodel_ex1))


.. parsed-literal::

    KerasModel Example1 created succesfully.


.. code:: ipython3

    # Defining the function to be charged
    def dncnn():
        x = layers.Input(shape=[None, None, 1])
        y = layers.Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same')(x)
        y = layers.Activation("relu")(y)

        # Middle layers: Conv + ReLU + BN
        for i in range(1, 16):
            y = layers.Conv2D(filters=64, kernel_size=5, strides=(1, 1), padding='same', use_bias=False)(y)
            y = layers.BatchNormalization(axis=-1, momentum=0.0, epsilon=1e-3)(y)
            y = layers.Activation("relu")(y)

        y = layers.Conv2D(filters=1, kernel_size=5, strides=(1, 1), use_bias=False, padding='same')(y)
        y = layers.Subtract()([x, y])

        # Keras model
        return models.Model(x, y)

.. code:: ipython3

    # Charging model into Example1
    kerasmodel_ex1.charge_model(model_function=dncnn)


.. parsed-literal::

    WARNING: Logging before flag parsing goes to stderr.
    W0902 10:47:13.272264 140353946789696 keras_model.py:119] You have loaded your model from a python function, which does not hold any information about weight values. Be sure to train the network before running your tests.


Notice the previous warning:

::

   W0821 09:12:14.533067 140116547233600 keras_model.py:118] You have loaded your model from a python function, which does not hold any information about weight values. Be sure to train the network before running your tests.

since you have loaded the model without any information about its
weights, we remark that you should run a training session before
performing inference.

Finally, it may be the case that your architecture has additional
parameters. Consider the following example,

.. code:: python

   def dncnn(depth=17, n_filters=64, kernel_size=(3, 3), n_channels=1):
       x = layers.Input(shape=[None, None, 1])
       y = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, strides=(1, 1), padding='same')(x)
       y = layers.Activation("relu")(y)

       # Middle layers: Conv + ReLU + BN
       for i in range(1, depth - 1):
           y = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, strides=(1, 1), padding='same', use_bias=False)(y)
           y = layers.BatchNormalization(axis=-1, momentum=0.0, epsilon=1e-3)(y)
           y = layers.Activation("relu")(y)

       y = layers.Conv2D(filters=1, kernel_size=kernel_size, strides=(1, 1), use_bias=False, padding='same')(y)
       y = layers.Subtract()([x, y])

       # Keras model
       return models.Model(x, y)

this corresponds to the same architecture as the previous DnCNN, except
that it has additional parameters, such as “depth”, “n_filters”,
“kernel_size” and “n_channels”. You can still pass these to the
charge_model function,

.. code:: ipython3

    def dncnn_opt_params(depth=17, n_filters=64, kernel_size=(3, 3), n_channels=1):
        x = layers.Input(shape=[None, None, n_channels])
        y = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, strides=(1, 1), padding='same')(x)
        y = layers.Activation("relu")(y)

        # Middle layers: Conv + ReLU + BN
        for i in range(1, depth - 1):
            y = layers.Conv2D(filters=n_filters, kernel_size=kernel_size, strides=(1, 1), padding='same', use_bias=False)(y)
            y = layers.BatchNormalization(axis=-1, momentum=0.0, epsilon=1e-3)(y)
            y = layers.Activation("relu")(y)

        y = layers.Conv2D(filters=1, kernel_size=kernel_size, strides=(1, 1), use_bias=False, padding='same')(y)
        y = layers.Subtract()([x, y])

        # Keras model
        return models.Model(x, y)

.. code:: ipython3

    kerasmodel_ex2 = model.KerasModel(model_name="Example2", logdir="../../logs/Keras")
    print("KerasModel {} created succesfully.".format(kerasmodel_ex2))
    kerasmodel_ex2.charge_model(model_function=dncnn_opt_params, depth=20, kernel_size=(7, 7), n_channels=3)


.. parsed-literal::

    KerasModel Example2 created succesfully.


.. parsed-literal::

    W0902 10:47:45.332327 140353946789696 keras_model.py:119] You have loaded your model from a python function, which does not hold any information about weight values. Be sure to train the network before running your tests.


From a file
-----------

There are two ways of charging a model from a file:

1. Charging an architecture (without weights) through a .json or .yaml
   file. This is done through a model previously saved using
   “keras.models.Model.to_json()” or “keras.models.Model.to_yaml()”
   method. As an example, consider the following:

.. code:: python

   net = dncnn()
   net.to_json("path_to_save_your_json_file")
   net.to_yaml("path_to_save_your_yaml_file")

In those two cases, the network is saved without any information about
its training or weights, so you should run a training session before
using your model for inference.

2. Charging the complete model (weights + architecture) using a
   .json/.yaml file + .hdf5 file, or only a .hdf5 file. Keras can save
   either only weights or weights + architecture into a .hdf5 file. That
   depends on the commands you have used, for instance,

.. code:: python

   net = dncnn()
   # Training of neural net
   net.save("model.hdf5") # This saves both weights and architecture.
   net.save_weights("weights.hdf5") # This saves only the weights.

To charge a model using a file, you simply need to pass it to
“**charge_model**” through the “model_path” parameters. An example is
shown bellow, on `Running Inference <#keras-running-inference>`__.

.. code:: ipython3

    # Frees memory
    kerasmodel_ex1 = None
    kerasmodel_ex2 = None
    gc.collect()




.. parsed-literal::

    290



.. code:: ipython3

    # Loads model from .hdf5 file.
    kerasmodel_ex3 = model.KerasModel(model_name="Inference_ex1", logdir="../../logs/Keras")
    kerasmodel_ex3.charge_model(model_path="../../pretrained_models/Keras/dncnn/model.hdf5")


.. parsed-literal::

    /home/efernand/repos/Summer_Internship_2019/venv/lib/python3.6/site-packages/keras/engine/saving.py:292: UserWarning: No training configuration found in save file: the model was *not* compiled. Compile it manually.
      warnings.warn('No training configuration found in save file: '


.. code:: ipython3

    # Get batch from valid_generator
    noisy_imgs, clean_imgs = next(valid_generator)
    # Performs inference on noisy images
    rest_imgs = kerasmodel_ex3(noisy_imgs)
    display_results(clean_imgs, noisy_imgs, rest_imgs, str(kerasmodel_ex3))



.. image:: Figures/output_24_0.png


.. code:: ipython3

    kerasmodel_ex3 = None
    gc.collect()




.. parsed-literal::

    25129



Training a KerasModel
---------------------

To run a training session, you only need to have a dataset, such as
defined in the DatasetUsage.ipynb. Once you created a DatasetGenerator
for your training images (and possibly, for you validation images) you
can call the “**train**” method from KerasModel class, which takes the
following parameters,

-  train_generator: any instance of a dataset generator class. This
   class will yield the data pairs.
-  valid_generator: optional. Specify it if you have validation data at
   hand.
-  n_epochs: number of training epochs. Default is 100.
-  n_stages: number of training batches drawn at random from the dataset
   at each training epoch. Default value is 500.
-  learning_rate: constant regulating the weight updates in your model.
   Default is 1e-3.
-  optimizer_name: you can specify the optimizer’s name for you model.
   You can do this by lookin at the names in `Keras
   documentation <https://keras.io/optimizers/>`__. Default is “Adam”
   optimizer.
-  metrics: list of metrics that will be tracked during training. There
   are a couple of useful metrics implemented on **evaluation** module
   (such as PSNR, SSIM, MSE) but you can also implement your own
   following `Keras conventions <https://keras.io/metrics/>`__.
-  kcallbacks: list of Keras callbacks. You can either use `Keras
   default callbacks <https://keras.io/callbacks/>`__ or the callbacks
   defined on :py:mod:`evaluation` module.
-  loss: A metric that will be used in optimization as the objective
   function to be minimized. You can either use `Keras
   default losses <https://keras.io/losses/>`__ or the metrics
   defined on :py:mod:`evaluation` module.
-  valid_steps: number of validation batches drawn at each validation
   epoch.

To show how a keras model can be trained, consider the training of a
DnCNN as stated on its `original
paper <https://arxiv.org/pdf/1608.03981.pdf>`__:

-  DnCNN for gaussian denoising has depth 17, n_filters 64, kernel_size
   (3, 3).
-  It is trained on :math:`40 \times 40` patches extracted from BSDS
   images, corrupted with fixed-variance gaussian noise
   (:math:`\sigma=25`, for instance).

For evaluation, we will use a disjoint subset of BSDS, consisting on 68
images which are not present in the training dataset.

.. code:: ipython3

    # KerasModel
    kerasmodel_ex4 = model.KerasModel(model_name="Example4", logdir="../../logs/Keras")
    print("KerasModel {} created succesfully.".format(kerasmodel_ex4))
    kerasmodel_ex4.charge_model(model_function=dncnn_opt_params, depth=17, kernel_size=(3, 3), n_channels=1)


.. parsed-literal::

    KerasModel Example4 created succesfully.


.. parsed-literal::

    W0902 10:51:00.440434 140353946789696 keras_model.py:119] You have loaded your model from a python function, which does not hold any information about weight values. Be sure to train the network before running your tests.


.. code:: ipython3

    kerasmodel_ex4.train(train_generator=train_generator,
                         valid_generator=valid_generator,
                         n_epochs=100,
                         n_stages=465,
                         learning_rate=1e-3,
                         optimizer_name="Adam",
                         metrics=[evaluation.DnCNNSchedule(),
                                  evaluation.CheckpointCallback(kerasmodel_ex4, monitor="val_PSNR"),
                                  evaluation.TensorboardImage(valid_generator, kerasmodel_ex4)],
                         loss=evaluation.mse,
                         valid_steps=10)

.. code:: ipython3

    kerasmodel_ex4 = None
    tf.reset_default_graph()
    gc.collect()