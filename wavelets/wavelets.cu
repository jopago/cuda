#include <cuda_runtime.h>
#include <common/utils.h>

#include "daubechies4.h"


/*  The Daubechies-4 wavelet forward pass
    I adapted this code from http://bearcave.com/misl/misl_tech/wavelets/index.html
    To compute the full the full wavelet transform of a signal of size N
    We call this kernel log_2(N) times (assuming N is power of 2) */

__global__ void gpu_dwt_pass(double *src, double *dest, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = n >> 1;

    if(2*i < (n-3)) {
        dest[i]             = src[2*i]*h[0] + src[2*i+1]*h[1] + src[2*i+2]*h[2] + src[2*i+3]*h[3];
        dest[i+half]        = src[2*i]*g[0] + src[2*i+1]*g[1] + src[2*i+2]*g[2] + src[2*i+3]*g[3];
    }
    if(2*i == (n-2)) {
        dest[i]         = src[n-2]*h[0] + src[n-1]*h[1] + src[0]*h[2] + src[1]*h[3];
        dest[i+half]    = src[n-2]*g[0] + src[n-1]*g[1] + src[0]*g[2] + src[1]*g[3];
    }

}


int gpu_dwt(double *t, int n)
{
    assert(check_power_two(n));

    size_t size = n*sizeof(double);
    double *d_src,*d_dst;

    int threadsPerBlock = 512;
    int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CALL(cudaMalloc((void**)&d_src,size));
    CUDA_CALL(cudaMalloc((void**)&d_dst,size));

    CUDA_CALL(cudaMemcpy(d_src,t,size,cudaMemcpyHostToDevice));

    clock_t begin, end;

    begin = clock();
    while(n >= 4)
    {
        gpu_dwt_pass<<<blocksPerGrid,threadsPerBlock>>>(d_src,d_dst,n);
        CUDA_CALL(cudaMemcpy(d_src,d_dst,size,cudaMemcpyDeviceToDevice));
        n = n>>1;
    }
    end = clock();
    CUDA_CALL(cudaMemcpy(t,d_src,size,cudaMemcpyDeviceToHost));
    
    printf("GPU Elapsed: %lfs \n", elapsed(begin,end));
    return 0;
}

void cpu_d4_transform(double *t, const int n)
{
    
    if (n >= 4) 
    {
        int i=0,j=0;
        const int half = n>>1;

        double * tmp = (double*)malloc(sizeof(double)*n);

        if(!tmp) 
        {
            fprintf(stderr, "cannot allocate memory for daubechies transform");
            exit(EXIT_FAILURE);
        }

        for (i = 0; i < half; i++) 
        {
            j = 2*i;
            if (j < n-3) {
                tmp[i]      = t[j]*_h[0] + t[j+1]*_h[1] + t[j+2]*_h[2] + t[j+3]*_h[3];
                tmp[i+half] = t[j]*_g[0] + t[j+1]*_g[1] + t[j+2]*_g[2] + t[j+3]*_g[3];
            } 
            else { 
                break; 
            }
        }

        tmp[i]      = t[n-2]*_h[0] + t[n-1]*_h[1] + t[0]*_h[2] + t[1]*_h[3];
        tmp[i+half] = t[n-2]*_g[0] + t[n-1]*_g[1] + t[0]*_g[2] + t[1]*_g[3];

        memcpy(t,tmp,n*sizeof(double));
        free(tmp);
    }
}

void cpu_d4_inv_transform(double *t, int n)
{
    if(n >= 4)
    {
        int i=0,j;
        int half = n >> 1;

        double * tmp = (double*)malloc(sizeof(double)*n);

        if(!tmp) 
        {
            fprintf(stderr, "cannot allocate memory for daubechies transform");
            exit(EXIT_FAILURE);
        }

       
        tmp[0] = t[half-1]*_ih[0] + t[n-1]*_ih[1] + t[0]*_ih[2] + t[half]*_ih[3];
        tmp[1] = t[half-1]*_ig[0] + t[n-1]*_ig[1] + t[0]*_ig[2] + t[half]*_ig[3];
        j = 2;
        for (;i < half-1; i++) 
        { 
          tmp[j++]    = t[i]*_ih[0] + t[i+half]*_ih[1] + t[i+1]*_ih[2] + t[i+half+1]*_ih[3];
          tmp[j++]    = t[i]*_ig[0] + t[i+half]*_ig[1] + t[i+1]*_ig[2] + t[i+half+1]*_ig[3];
        }

        memcpy(t,tmp,n*sizeof(double));
        free(tmp);
    }
}



void cpu_dwt(double* t, int N)
{
    assert(check_power_two(N));
    int n=N;

    while(n >= 4) 
    {
        cpu_d4_transform(t,n);
        n >>= 1;
    }
}

void cpu_idwt(double *t, int N)
{
    assert(check_power_two(N));

    int n;

    for(n = 4; n <= N; n <<= 1)
    {
        cpu_d4_inv_transform(t,n);
    }
}

int main()
{
    const int N = (1<<16);
    size_t size = N*sizeof(double);
    clock_t begin, end; 
    double * gpu_coef   = (double*)malloc(size);
    double * cpu_coef   = (double*)malloc(size);
    double * x0         = (double*)malloc(size);

    if(!gpu_coef || !cpu_coef || !x0) {
        fprintf(stderr, "%s\n", "could not allocate memory for signals!\n");
        exit(EXIT_FAILURE);
    }

    fill_rand(x0, N);
    memcpy(gpu_coef,x0,size);
    memcpy(cpu_coef,x0,size); // save initial (random) array 

    gpu_dwt(gpu_coef,N);

    begin = clock();
    cpu_dwt(cpu_coef,N);
    end = clock();

    printf("CPU elapsed: %lfs\n", elapsed(begin,end));

    /* Test wavelet decomposition */

    if(test_arrays_equal(gpu_coef,cpu_coef,N)) 
    {
        printf("Wavelet decomposition is the same on CPU and GPU.\n");
    } else {
        printf("Wavelet decompsition is not the same on CPU and GPU!\n");
        exit(EXIT_FAILURE);
    }

    cpu_idwt(cpu_coef,N);

    if(test_arrays_equal(cpu_coef,x0,N)) 
    {
        printf("Inverse wavelet transform on CPU: success!\n");
    }
}