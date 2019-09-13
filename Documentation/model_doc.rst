model module
============

This module contains classes used for wrapping denoising algorithms in order to provide common functionality and behavior.
The frst distinction we make is between deep learning based denoising algorithms, and filtering algorithms. The functionality
of these two models is standardized by :class:`model.AbstractDenoiser`.

Deep Learning models, however, are based on frameworks such as `Keras
<https://keras.io>`_ and `Pytorch
<https://pytorch.org/>`_ which differ in syntax. To standardize the behavior of Deep Learning models across frameworks,
we propose the
:class:`model.AbstractDeepLearningModel` interface, which unifies three kinds of functionalities that all deep learning models
should provide:

1. charge_model, which builds the computational graph of the network.
2. train, which trains the network on data.
3. inference, which denoises data.

To give a better idea of the class structure and its relations to the other modules, we show the following UML class diagram,

.. image:: /Figures/ClassDiagram.png
    :alt:

The documentation of this module is divided as follows,

1. **Interface Classes:** Documents the two main abstract classes which are AbstractDenoiser and AbstractDeepLearningModel.
2. **Wrapper Classes:** Documents the 6 wrapper classes in the benchmark, which are FilteringModel, KerasModel,
   PytorchModel, TfModel, MatlabModel, OnnxModel, MatconvnetModel.
3. **Built-in Architectures:** Documents the neural network architectures already provided by the benchmark.
4. **Filtering functions:** Documents the filtering functions already provided by the benchmark.
5. **Model utilities:** Documents functions for model conversion and graph editing.

Denoising Models
----------------
Interface Classes
^^^^^^^^^^^^^^^^^
.. automodule:: model
  :members: AbstractDenoiser, AbstractDeepLearningModel
  :special-members:
  :exclude-members: __dict__,__weakref__, __module__
  :show-inheritance:

Wrapper Classes
^^^^^^^^^^^^^^^
.. automodule:: model
  :noindex:
  :members: FilteringModel, KerasModel, PytorchModel, TfModel, MatlabModel, OnnxModel, MatconvnetModel
  :special-members:
  :exclude-members: __dict__,__weakref__, __module__
  :show-inheritance:


Built-in Architectures
----------------------
Keras Architectures
^^^^^^^^^^^^^^^^^^^
.. automodule:: model.architectures.keras
  :members: dncnn, rednet, mwcnn_1

Tensorflow Architectures
^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: model.architectures.tensorflow
  :members: dncnn

Pytorch Architectures
^^^^^^^^^^^^^^^^^^^^^
.. automodule:: model.architectures.pytorch
  :members: DnCNN
  :special-members:

Filtering functions
-------------------
Matlab-based functions
^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: model.filtering
  :members: BM3D

Model utilities
---------------
.. automodule:: model.utils
  :members: pb2onnx, freeze_tf_graph, onnx_dynamic_shapes
