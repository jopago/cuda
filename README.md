# Cuda kernels

Just some CUDA kernels I'm working on. To compile just use the makefile with `make all` (make sure to 
have `nvcc` added to your path and a working CUDA install.)

## 1D Convolution 

Simple 1D convolution Cuda kernel, benchmarking against CPU.

![](https://github.com/jopago/cuda/raw/master/conv1d/img/timing_conv1d.png)


## Wavelets 

A CUDA implementation of Discrete Wavelet Transform with Daubechies-4 wavelets. 

![](https://github.com/jopago/cuda/raw/master/wavelets/img/timing_wavelets.png)
