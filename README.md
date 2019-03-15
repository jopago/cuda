# CUDA

> This repo contains some CUDA kernels I'm working on for academic purposes, mostly about signal processing. To compile just use the makefile with `make all` (make sure to have `nvcc` added to your path and a working CUDA install).

## Live Haar wavelet decomposition with OpenCV

A CUDA implementation of 2D Haar Wavelet transform. I use it to compute the full-level decomposition of my webcam video feed with OpenCV (in colors with channels treated separately), thanks to GPU acceleration it can run smoothly. Here is a screenshot:

![](https://github.com/jopago/cuda/raw/master/haar/img/decomposition_frame.png)

## 2D Convolution 

A parallel 2D implementation of convolution with CUDA, and benchmarking against CPU. This task is highly parallel and the achieved speedup important (up to 50x with my settings), image displaying and reading is doen with OpenCV.

![](https://github.com/jopago/cuda/raw/master/conv2d/img/lena.png)
![](https://github.com/jopago/cuda/raw/master/conv2d/img/convolution_gpu.png)
![](https://github.com/jopago/cuda/raw/master/conv2d/img/convolution_gpu_sharpen.png)

![](https://github.com/jopago/cuda/raw/master/conv2d/img/timing_conv2d.png)

## Daubechies-4 Wavelets 

A CUDA implementation of Discrete Wavelet Transform with Daubechies-4 wavelets. Since this DWT algorithm is recursive, the speedup is less important, 4 times with my settings. 

![](https://github.com/jopago/cuda/raw/master/daubechies4/img/timing_wavelets.png)
