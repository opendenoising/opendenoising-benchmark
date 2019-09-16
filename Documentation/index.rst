OpenDenoising: An Open Benchmark for Image Restoration Methods
==============================================================

OpenDenoising is an open-source benchmark for comparing the performance of denoising algorithms. We currently support
generic denoising functions through Python and Matlab, as well as deep learning based denoisers through frameworks. The
table bellow shows the compatibility between our benchmark and various frameworks,

+----------------------+-----------------------+----------+-----------+
| Programming Language |       Framework       | Training | Inference |
+----------------------+-----------------------+----------+-----------+
|        Matlab        |       Matconvnet      |          |     x     |
|                      +-----------------------+----------+-----------+
|                      | Deep Learning Toolbox |     x    |     x     |
+----------------------+-----------------------+----------+-----------+
|        Python        |         Keras         |     x    |     x     |
|                      +-----------------------+----------+-----------+
|                      |       Tensorflow      |     x    |     x     |
|                      +-----------------------+----------+-----------+
|                      |        Pytorch        |     x    |     x     |
|                      +-----------------------+----------+-----------+
|                      |          ONNX         |          |     x     |
+----------------------+-----------------------+----------+-----------+

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   source/installation.rst
   source/api_doc.rst
   source/examples.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
