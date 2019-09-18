data module
===========

This module extends `keras.utils.Sequence
<https://keras.io/utils/>`_. dataset generators for the three main cases in image denoising:

1. **Clean Dataset:** Only clean (ground-truth) images are available. Hence, you will need to specify a function to
   corrupt the clean images, so that you can train your network with pairs (noisy, clean). On the section Artificial
   Noises, we cover built-in functions for adding noise to clean images.
2. **Full Dataset:** Both clean and noisy images are available. In that case, the dataset yields the image paris (clean,
   noisy).
3. **Blind Dataset:** Only noisy images are available. These datasets can be used for qualitative avaliation.

We remark that these classes can be used for training and inference on our Benchmark. It is also noteworthy that MatlabModel
objects, which are based on Matlab Deep Learning Toolbox, need to use the MatlabDatasetWrapper to be efficiently trained.
For more information, look at :class:`data.MatlabDatasetWrapper` and :class:`model.MatlabModel` classes documentation.

Along with these classes, we also provide built-in functionalities for preprocessing, such as Data Augmentation and
patch extraction. These are covered in the **Preprocessing Functions** section.

Data generation
---------------

.. automodule:: data
  :members: AbstractDatasetGenerator, BlindDatasetGenerator, CleanDatasetGenerator, FullDatasetGenerator, MatlabDatasetWrapper
  :special-members:
  :exclude-members: __dict__,__weakref__, __module__
  :show-inheritance:


Artificial Noises
-----------------
.. automodule:: data
  :members: gaussian_noise, poisson_noise, salt_and_pepper_noise, speckle_noise, super_resolution_noise
  :special-members:
  :exclude-members: __dict__,__weakref__, __module__
  :show-inheritance:


Preprocessing Functions
-----------------------
.. automodule:: data
  :members: dncnn_augmentation, gen_patches
  :special-members:
  :exclude-members: __dict__,__weakref__, __module__
  :show-inheritance:
