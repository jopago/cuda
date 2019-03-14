#include <cuda_runtime.h>
#include <common/utils.h>
#include <common/cuda_ptr.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "filters2d.h"

template <typename T>
void conv2d(T* in, int size_in, double* filter, int size_filter,
	T* out)
{

	int i,j,k;
	int radius = size_filter / 2;

	for(i=0;i<size_in;i++)
	{
		for(j=0;j<size_in;j++)
		{
			T sum = 0.0;
			int i_top_left = i-radius;
			int j_top_left = j-radius;

			// stride 1 with zero padding, the filte is centered at (i,j) 

			// 		a 	b 	c 
			// 		d 	e 	f -> filter[0]*a + filter[1]*b + ... + filter[size_filter-2]*h + filter[size_filter-1]*i
			// 		g 	h 	i 

			for(k=0;k<size_filter*size_filter;k++)
			{
				// receptive field of the convolution 

				int _i = i_top_left + (k/size_filter); 
				int _j = j_top_left + (k%size_filter); 

				// if one of the indices exceed the boundaries we don't accumulate the sum, i.e we pad with zeros
				if( (_i >= 0) && (_i < size_in) && (_j >= 0) && (_j < size_in))
				{
					int idx = _i * size_in + _j;
					sum += filter[k]*in[idx];
				}
			}

			out[i*size_in+j] = sum;
		}
	}
}

template <typename T>
__global__ void gpu_conv2d(T* in, int size_in, double *filter, int size_filter,
	double *out)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int radius = size_filter / 2;

	if((i < size_in) && (j < size_in))
	{
		double sum = 0.;
		int k;

		int i_top_left = i-radius;
		int j_top_left = j-radius;

		for(k=0;k<size_filter*size_filter;k++)
		{
			int _i = i_top_left + (k/size_filter); 
			int _j = j_top_left + (k%size_filter); 

			if( (_i >= 0) && (_i < size_in) && (_j >= 0) && (_j < size_in))
			{
				int idx = _i * size_in + _j;
				sum += filter[k]*in[idx];
			}
		}

		out[i*size_in+j] = sum;
	}
}

int timing()
{
	srand(123); 
	int N;
	FILE *csv = fopen("results/timing.csv", "w+");
	if(!csv)
	{
		fprintf(stderr, "%s\n", "(host) cannot open file results/timing.csv\n");
		exit(EXIT_FAILURE);
	}
	fprintf(csv, "%s,%s,%s\n", "N","CPU_Time","GPU_Time");

	for(N=256; N<=4096; N += 256)
	{
		int size_filter = 3;
		int size = N*N*sizeof(double);

		double *img 		= (double*)malloc(size);
		double *conv_gpu 	= (double*)malloc(size);
		double *conv_cpu 	= (double*)malloc(size);

		double cpu_time,gpu_time;
		clock_t begin,end;

		if(!img || !conv_gpu || !conv_cpu)
		{
			printf("(host) cannot allocate memory for images..\n");
			exit(EXIT_FAILURE);
		}

		fill_rand_2d(img,N);

		cuda_ptr<double> img_gpu(img, size), filter_gpu(fd_x,3*3*sizeof(double)),
			out_gpu(size);

		int blockWidth = 8; // 64 threads (pixels) per block

		dim3 dimBlock(blockWidth,blockWidth);
		dim3 dimGrid(N / dimBlock.x, N / dimBlock.y); 

		begin = clock();
		gpu_conv2d<<<dimGrid,dimBlock>>>(img_gpu.devptr(),N,filter_gpu.devptr(),size_filter,out_gpu.devptr());
		CUDA_CALL(cudaDeviceSynchronize());
		end = clock();

		gpu_time = elapsed(begin,end);

		out_gpu.to_host(conv_gpu,size);

		begin = clock();
		conv2d(img,N,fd_x,size_filter,conv_cpu);
		end = clock();

		cpu_time = elapsed(begin,end);

		printf("N: %d\n", N);
		printf("GPU Time: %lfs\n", gpu_time);
		printf("CPU Time: %lfs\n", cpu_time);

		if(test_arrays_equal(conv_cpu,conv_gpu,N*N))
		{
			printf("Test passed!\n");
		} else 
		{
			printf("Test failed!\n");
			exit(EXIT_FAILURE);
		}

		fprintf(csv, "%d,%lf,%lf\n", N, cpu_time, gpu_time);

		free(img);
		free(conv_cpu);
		free(conv_gpu);
	}

	return 1;
}

int main(int argc, char **argv)
{
	// srand(time(0));
    cv::Mat image = cv::imread("img/lena.png", 0); // grayscale 
    const int N = image.rows;
    const int size 	= N*N*sizeof(double);

    const int blockWidth 	= 8; // 64 threads (pixels) per block
    const int size_filter 	= 3;

    if(!image.data)
    {
        std::cout <<  "(host) could not open or find the image" << std::endl ;
        exit(EXIT_FAILURE);
    }

    double *img 		= new double[N*N];
    double *conv 		= new double[N*N];
    double *conv_gpu 	= new double[N*N];

    int i,j;

    for(i=0;i<N;i++)
    {
    	for(j=0;j<N;j++)
    	{
    		img[i*N + j] = image.at<unsigned char>(j,i);
    	}
    }

    cuda_ptr<double> img_gpu(img,size);
    cuda_ptr<double> filter_gpu(laplace2d,size_filter*size_filter*sizeof(double));
    cuda_ptr<double> out_gpu(img,size);

    clock_t begin, end;

	dim3 dimBlock(blockWidth,blockWidth);
	dim3 dimGrid(N / dimBlock.x, N / dimBlock.y); 

	begin = clock();
	gpu_conv2d<<<dimGrid,dimBlock>>>(img_gpu.devptr(),N,filter_gpu.devptr(),size_filter,out_gpu.devptr());
	CUDA_CALL(cudaDeviceSynchronize());
	end = clock();

	std::cout << "GPU Time: " << elapsed(begin, end) << "s\n";

	begin = clock();
    conv2d<double>(img, N, laplace2d, size_filter, conv);
    end = clock();

    cv::namedWindow("Lena", cv::WINDOW_AUTOSIZE);
    cv::imshow("Lena", image);

    std::cout << "CPU Time: " << elapsed(begin, end) << "s\n";

    // display CPU convolution

    for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			image.at<unsigned char>(j,i) = max(0.0,conv[i*N + j]);
		}
	}

    cv::namedWindow("CPU", cv::WINDOW_AUTOSIZE);
    cv::imshow("CPU",image);

    // copy result from gpu 
	out_gpu.to_host(conv_gpu, size);

	// display GPU convolution
    for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			image.at<unsigned char>(j,i) = max(0.0,conv_gpu[i*N + j]);
		}
	}

    cv::namedWindow("GPU", cv::WINDOW_AUTOSIZE);
    cv::imshow("GPU",image);
    cv::imwrite("img/convolution_gpu.png",image);

    cv::waitKey(0);

    delete img;
    delete conv;
    delete conv_gpu; 

	return 1;
}