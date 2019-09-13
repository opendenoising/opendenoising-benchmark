# Model Module

This model contains classes for wrapping Deep Learning frameworks and denoising algorithm functions. In "./architectures"
you may find a list of built-in neural network architectures. In filtering, there is a list of build-in functions for
performing denoising.

## Built-in architectures

### Keras

* DnCNN<sup>1</sup>
* xDnCNN<sup>2</sup>
* REDNet<sup>3</sup>
* MWCNN<sup>4, 5</sup>

### Tensorflow

* DnCNN<sup>1</sup>

### Pytorch

* DnCNN<sup>1</sup>
* xDnCNN<sup>2</sup>

### Matlab

* DnCNN<sup>1</sup>

## Built-in filtering algorithms

__Note:__ This sub-module contains third-party software. If you do use it you automatically agreeing with author's
license terms.

* BM3D<sup>6</sup> [license terms](http://www.cs.tut.fi/~foi/GCF-BM3D/legal_notice.html)

# References
--

1. Zhang K, Zuo W, Chen Y, Meng D, Zhang L. Beyond a gaussian denoiser: Residual learning of deep cnn for image
   denoising. IEEE Transactions on Image Processing. 2017 <a name="dncnn"></a>
2. Kligvasser I, Rott Shaham T, Michaeli T. xUnit: Learning a spatial activation function for efficient image
   restoration. InProceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2018
3. Mao XJ, Shen C, Yang YB. Image restoration using convolutional auto-encoders with symmetric skip connections.
   arXiv preprint arXiv:1606.08921. 2016
4. Liu P, Zhang H, Lian W, Zuo W. Multi-Level Wavelet Convolutional Neural Networks. IEEE Access. 2019
5. Liu P, Zhang H, Zhang K, Lin L, Zuo W. Multi-level wavelet-CNN for image restoration. InProceedings of the IEEE
   Conference on Computer Vision and Pattern Recognition Workshops 2018
6. Dabov K, Foi A, Katkovnik V, Egiazarian K. Image denoising by sparse 3-D transform-domain collaborative
   filtering. IEEE Transactions on image processing. 2007