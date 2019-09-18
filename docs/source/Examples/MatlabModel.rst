Matlab Model tutorial
=====================

This tutorial is a part of Model module guide. Here, we explore how you
can use the MatlabModel wrapper to use your Matlab deep learning models
in the benchmark.

.. code:: python

    # Python packages
    import gc
    import numpy as np
    import matlab.engine
    import matplotlib.pyplot as plt

    from functools import partial
    from OpenDenoising import data
    from OpenDenoising import model
    from OpenDenoising import evaluation

    eng = matlab.engine.start_matlab()

For now on, we suppose you are running your codes on the project root folder.

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

.. code-block:: python

    # Validation images generator
    valid_generator = data.DatasetFactory.create(path="./tmp/BSDS500/Valid",
                                                 batch_size=8,
                                                 n_channels=1,
                                                 noise_config={data.utils.gaussian_noise: [25]},
                                                 name="BSDS_Valid")


Charging a model
-----------------

To charge a Matlab model, you can either specify a Matlab function or a
.mat file containing the architecture you want to train/test. In both
cases, you need to specify a string.

From a file
-----------

To charge a model from a file, you need to specify the path to the .mat
file containing the model’s architecture. Notice that for models that
predict the residual, rather than the restored image, “**return_diff**”
should be specified as True,

.. code:: python

    matlabModel = model.MatlabModel(return_diff=True)

.. code:: python

    matlabModel.charge_model(model_path="./Additional Files/Matlab Models/dncnn_matlab.mat")

After charging the model into the wrapper object, the network object will be available on Matlab’s workspace. The
following command prints the workspace:

.. code:: python

    print(matlabModel.engine.workspace)




.. parsed-literal::

  Matlab's Workspace:
  Name                   Size              Bytes  Class                                 Attributes

  layers                50x1             4465192  nnet.cnn.layer.Layer





From a Function
~~~~~~~~~~~~~~~

To specify a model from a function, you need to specify the path to the
.m file that has the function that will build your model. This string is
used internally to add the .m file to the path, and the to call the
function using Matlab’s engine.

**Note:** you may still pass extra arguments through kwargs, as if they
were going to feed a normal Python function.

.. code:: python

    matlabModel2 = model.MatlabModel(return_diff=True)
    matlabModel2.charge_model(model_function="./OpenDenoising/model/architectures/matlab/dncnn.m")

Inference with MatlabModel
---------------------------

To perform inference, you may use the “\__call_\_” method in MatlabModel
class. This method uses the Matlab’s engine to internally call
“denoiseImage” matlab function, that uses the network object to denoise
an input batch.

.. code:: python

    # Get batch from valid_generator
    noisy_imgs, clean_imgs = next(valid_generator)
    # Performs inference on noisy images
    rest_imgs = matlabModel(noisy_imgs)

.. code:: python

    display_results(clean_imgs, noisy_imgs, noisy_imgs - rest_imgs, str(matlabModel))



.. image:: Figures/matlab_output_18_0.png


Training a MatlabModel
-----------------------

To train a MatlabModel, you need to specify a training (and possibly a
validation) dataset through a string. This string correspond to the name
of the dataset in Matlab’s workspace.

To create the dataset in the workspace, you can use the classes
‘imageDatastore’, ‘CleanMatlabDataset’ and ‘FullMatlabDataset’, which
are Matlab classes for generating data to train Deep Learning models.

Using a CleanDataset
~~~~~~~~~~~~~~~~~~~~

As in the case of Python’s CleanDatasetGenerator, to specify a Clean
Dataset using Matlab you need to specify the noising function, called
noiseFcn. This function should be specified as a string, that has the
**lambda signature** on it.

For instance, if you want to use Gaussia noise on your dataset, you need
to specify:

noiseFcn = “@(I) imnoise(I, ‘gaussian’, 0, 25/255)”.

For more complex kinds of functions, you can implement it as a .m
function, and specify its arguments via the same strategy.

**Note:** You should make sure that “./OpenDenoising/data/” folder is on
Matlab’s path (add it to pathdef.m).

.. code:: python

    dataset_train_wrapper = data.MatlabCleanDatasetGenerator(matlabModel2.engine, images_path="./tmp/BSDS500/Train/ref",
                                                             partition="Train")
    dataset_train_wrapper()

.. code:: python

    dataset_valid_wrapper = data.MatlabCleanDatasetGenerator(matlabModel2.engine, images_path="./tmp/BSDS500/Valid/ref",
                                                             partition="Valid")
    dataset_valid_wrapper()

.. code:: python

    print(matlabModel2.engine.workspace)




.. parsed-literal::


  Matlab's Workspace:
  Name                   Size              Bytes  Class                                 Attributes

  ME                     1x1                1138  MException
  imds_Train             1x1                   8  matlab.io.datastore.ImageDatastore
  imds_Train_noise       1x1                   8  CleanMatlabDataset
  imds_Valid             1x1                   8  matlab.io.datastore.ImageDatastore
  imds_Valid_noise       1x1                   8  CleanMatlabDataset
  layers                50x1             4465192  nnet.cnn.layer.Layer



.. code:: python

    matlabModel2.train(train_generator="imds_Train_noise", valid_generator="imds_Valid_noise")
