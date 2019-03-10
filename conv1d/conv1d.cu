#include <cuda_runtime.h>
#include <common/utils.h>

#include "filters.h"

void conv1d(double * in, const int size_in, double * filter, const int size_filter, 
            double *out)
{
    int i,j;
    int radius = size_filter / 2;
    for(i=0;i<size_in;i++)
    {
        double sum = 0.0;

        for(j=0;j<=radius;j++)
        {
            if( (i-j) >= 0) // left
            {
                sum += filter[radius - j]*in[i-j];
            }

            if( (i+j) < size_in && (j != 0)) // right 
            {
                sum += filter[radius + j]*in[i+j];
            }
        }

        out[i] = sum; 
    }
}

__global__ void gpu_conv1d(double *in, const int size_in, double * filter, const int size_filter,
                            double *out)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x; 

    if(i < size_in)
    {
        double sum = 0.0;
        int radius = size_filter / 2;
        int j;

        for(j=0;j<=radius;j++)
        {
            if( (i-j) >= 0) // left
            {
                sum += filter[radius - j]*in[i-j];
            }

            if( (i+j) < size_in && (j != 0)) // right 
            {
                sum += filter[radius + j]*in[i+j];
            }
        }

        out[i] = sum; 
    }
}

int test_conv1d(int N, double& cpu_time, double& gpu_time)
{
    double * signal = (double*)malloc(N*sizeof(double));
    double * result = (double*)malloc(N*sizeof(double));
    double * gpu_result = (double*)malloc(N*sizeof(double));
    clock_t begin, end;

    double *d_signal,*d_result,*d_filter;

    CUDA_CALL(cudaMalloc((void**)&d_signal, N*sizeof(double)));
    CUDA_CALL(cudaMalloc((void**)&d_result, N*sizeof(double)));


    fill_rand(signal,N);

    int fw = 5;
    double * filter = ones(fw);

    CUDA_CALL(cudaMalloc((void**)&d_filter, fw*sizeof(double)));

    CUDA_CALL(cudaMemcpy(d_signal, signal, N*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_filter, filter, fw*sizeof(double), cudaMemcpyHostToDevice));
    
    begin = clock();
    conv1d(signal, N, filter, fw, result);
    end = clock();

    printf("CPU elapsed: %lfs\n", elapsed(begin,end));
    cpu_time = elapsed(begin,end);

    int threadsPerBlock = 512;
    int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;

    begin = clock();
    gpu_conv1d<<<blocksPerGrid,threadsPerBlock>>>(d_signal, N, d_filter, fw, d_result);
    cudaDeviceSynchronize();
    end = clock();

    printf("GPU elapsed: %lfs\n", elapsed(begin,end));
    gpu_time = elapsed(begin,end);

    CUDA_CALL(cudaMemcpy(gpu_result, d_result, N*sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(d_signal));
    CUDA_CALL(cudaFree(d_result));
    CUDA_CALL(cudaFree(d_filter));

    free(signal);

    if(!test_arrays_equal(gpu_result,result,N))
    {
        printf("Test failed!\n");

        free(result);
        free(gpu_result);
        return 0;
    }

    printf("Test passed!\n");
   
    free(result);
    free(gpu_result);
    return 1;
}

void timing()
{
    int N = 10000;
    double cpu,gpu;
    FILE *csv;

    csv = fopen("results/timing.csv", "w+");
    if(!csv)
    {
        fprintf(stderr, "(host) unable to create timing results file!\n");
        exit(EXIT_FAILURE);
    }

    fprintf(csv, "%s,%s,%s\n", "N", "CPU_Time", "GPU_Time");

    for(N=1e4;N<=5e7;N *= 2)
    {
        if(!test_conv1d(N,cpu,gpu)) exit(EXIT_FAILURE);

        fprintf(csv, "%d,%lf,%lf\n", N, cpu, gpu);
    }

    fclose(csv);
}

int main(int argc, char** argv)
{
    // srand(time(0));
    
    timing();
    return 0;
}