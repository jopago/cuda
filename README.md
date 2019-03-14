# Cuda kernels

Just some CUDA kernels I'm working on. To compile just use the makefile with `make all` (make sure to 
have `nvcc` added to your path and a working CUDA install).


## 2D Convolution 

Simple 2D convolution Cuda kernel, benchmarking against CPU. This task is highly parallel and the achieved speedup important
(up to 50x with my settings). 


![](https://github.com/jopago/cuda/raw/master/conv2d/img/lena.png)
![](https://github.com/jopago/cuda/raw/master/conv2d/img/convolution_gpu.png)
![](https://github.com/jopago/cuda/raw/master/conv2d/img/convolution_gpu_sharpen.png)

![](https://github.com/jopago/cuda/raw/master/conv2d/img/timing_conv2d.png)

## Wavelets 

A CUDA implementation of Discrete Wavelet Transform with Daubechies-4 wavelets. Since this DWT algorithm is recursive, the speedup is less important, 4 times with my settings. 

![](https://github.com/jopago/cuda/raw/master/wavelets/img/timing_wavelets.png)
