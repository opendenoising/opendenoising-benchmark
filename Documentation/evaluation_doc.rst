evaluation module
=======================

The evaluation module was designed to provide functions that implement numerical metrics and visualisations during training
and inference, as well as callbacks.

Callbacks
---------
.. automodule:: evaluation
  :members: TensorboardImage, LrSchedulerCallback, DnCNNSchedule, StepSchedule, PolynomialSchedule, ExponentialSchedule, CheckpointCallback
  :special-members:
  :show-inheritance:

Metrics
-------

.. automodule:: evaluation
  :members: Metric
  :special-members:
  :show-inheritance:

Tensorflow Metrics
^^^^^^^^^^^^^^^^^^
.. automodule:: evaluation
  :members: tf_ssim, tf_mse, tf_psnr, tf_se
  :special-members:
  :show-inheritance:

Skimage Metrics
^^^^^^^^^^^^^^^
.. automodule:: evaluation
  :members: skimage_ssim, skimage_mse, skimage_psnr
  :special-members:
  :show-inheritance:

Visualisations
--------------
.. automodule:: evaluation
  :members: Visualisation
  :special-members:
  :show-inheritance:

Functions
^^^^^^^^^
.. automodule:: evaluation
  :members: boxplot
  :special-members:
  :show-inheritance:




