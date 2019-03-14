#include <cuda_runtime.h>
#include <common/utils.h>

#include "filters2d.h"

void conv2d(double *in, int size_in, double *filter, int size_filter,
	double *out)
{

	int i,j,k;
	int radius = size_filter / 2;

	for(i=0;i<size_in;i++)
	{
		for(j=0;j<size_in;j++)
		{
			double sum = 0.0;
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

__global__ void gpu_conv2d(double *in, int size_in, double *filter, int size_filter,
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

int main(int argc, char **argv)
{
	// srand(time(0));
	srand(123); 

	int N = 1024, size_filter = 3;
	int size = N*N*sizeof(double);

	double *img = (double*)malloc(size);
	double *conv_gpu = (double*)malloc(size);
	double *conv_cpu = (double*)malloc(size);

	double *img_gpu,*filter_gpu,*out_gpu;
	double cpu_time,gpu_time;
	clock_t begin,end;

	if(!img || !conv_gpu || !conv_cpu)
	{
		printf("(host) cannot allocate memory for images..\n");
		exit(EXIT_FAILURE);
	}

	fill_rand_2d(img,N);

	CUDA_CALL(cudaMalloc(&img_gpu, size));
	CUDA_CALL(cudaMalloc(&filter_gpu,size_filter*size_filter*sizeof(double)));
	CUDA_CALL(cudaMalloc(&out_gpu,size));

	CUDA_CALL(cudaMemcpy(img_gpu, img, size, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(filter_gpu, fd_x, size_filter*size_filter*sizeof(double), cudaMemcpyHostToDevice));

	int blockWidth = 16;

	dim3 dimBlock(blockWidth,blockWidth); // 256 threads per block
	dim3 dimGrid(N / dimBlock.x, N / dimBlock.y); 

	begin = clock();
	gpu_conv2d<<<dimGrid,dimBlock>>>(img_gpu,N,filter_gpu,size_filter,out_gpu);
	CUDA_CALL(cudaDeviceSynchronize());
	end = clock();

	gpu_time = elapsed(begin,end);

	CUDA_CALL(cudaMemcpy(conv_gpu,out_gpu,size,cudaMemcpyDeviceToHost));

	CUDA_CALL(cudaFree(out_gpu));
	CUDA_CALL(cudaFree(img_gpu));
	CUDA_CALL(cudaFree(filter_gpu));

	begin = clock();
	conv2d(img,N,fd_x,size_filter,conv_cpu);
	end = clock();

	cpu_time = elapsed(begin,end);

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

	free(img);
	free(conv_cpu);
	free(conv_gpu);

	return 1;
}