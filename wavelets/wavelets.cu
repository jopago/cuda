#include <cuda_runtime.h>
#include <common/utils.h>

#include "daubechies4.h"

void cpu_d4_transform(double *src, double* dest, const int n)
{
    
    if (n >= 4) 
    {
        int i=0,j=0;
        const int half = n>>1;

        for (i = 0; i < half; i++) 
        {
            j = 2*i;
            if (j < n-3) {
                dest[i]      = src[j]*_h[0] + src[j+1]*_h[1] + src[j+2]*_h[2] + src[j+3]*_h[3];
                dest[i+half] = src[j]*_g[0] + src[j+1]*_g[1] + src[j+2]*_g[2] + src[j+3]*_g[3];
            } 
            else { 
                break; 
            }
        }

        dest[i]      = src[n-2]*_h[0] + src[n-1]*_h[1] + src[0]*_h[2] + src[1]*_h[3];
        dest[i+half] = src[n-2]*_g[0] + src[n-1]*_g[1] + src[0]*_g[2] + src[1]*_g[3];
    }
}

void cpu_d4_inv_transform(double *src, double *dest, int n)
{

    if(n >= 4)
    {
        int i=0,j;
        int half = n >> 1;
       
        dest[0] = src[half-1]*_ih[0] + src[n-1]*_ih[1] + src[0]*_ih[2] + src[half]*_ih[3];
        dest[1] = src[half-1]*_ig[0] + src[n-1]*_ig[1] + src[0]*_ig[2] + src[half]*_ig[3];
        j = 2;
        for (;i < half-1; i++) 
        { 
          dest[j++]    = src[i]*_ih[0] + src[i+half]*_ih[1] + src[i+1]*_ih[2] + src[i+half+1]*_ih[3];
          dest[j++]    = src[i]*_ig[0] + src[i+half]*_ig[1] + src[i+1]*_ig[2] + src[i+half+1]*_ig[3];
        }
    }
}

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

__global__ void gpu_idwt_pass(double *src, double *dest, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = n >> 1;

    if(i == 0)
    {
        dest[0] = src[half-1]*ih[0] + src[n-1]*ih[1] + src[0]*ih[2] + src[half]*ih[3];
        dest[1] = src[half-1]*ig[0] + src[n-1]*ig[1] + src[0]*ig[2] + src[half]*ig[3];
    } 
    if (i < (half-1)) 
    {
        dest[2*i+2]    = src[i]*ih[0] + src[i+half]*ih[1] + src[i+1]*ih[2] + src[i+half+1]*ih[3];
        dest[2*i+3]    = src[i]*ig[0] + src[i+half]*ig[1] + src[i+1]*ig[2] + src[i+half+1]*ig[3];
    }
}

double gpu_dwt(double *t, int N)
{
    assert(check_power_two(N));

    size_t size = N*sizeof(double);
    double *d_src,*d_dst;
    int n = N;

    int threadsPerBlock = 512;
    int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CALL(cudaMalloc((void**)&d_src,size));
    CUDA_CALL(cudaMalloc((void**)&d_dst,size));

    CUDA_CALL(cudaMemcpy(d_src,t,size,cudaMemcpyHostToDevice));

    clock_t begin, end;

    begin = clock();
    while(n >= 4)
    {
        gpu_dwt_pass<<<blocksPerGrid,threadsPerBlock>>>(d_src,d_dst,n);
        // we need only copy the n first elements, not the whole signal
        CUDA_CALL(cudaMemcpy(d_src,d_dst,n*sizeof(double),cudaMemcpyDeviceToDevice)); 
        n = n>>1;
    }
    cudaDeviceSynchronize();
    end = clock();
    CUDA_CALL(cudaMemcpy(t,d_src,size,cudaMemcpyDeviceToHost));
    
    printf("GPU Elapsed: %lfs \n", elapsed(begin,end));

    CUDA_CALL(cudaFree(d_src));
    CUDA_CALL(cudaFree(d_dst));
    return elapsed(begin,end);
}

double gpu_idwt(double *t, int N)
{
    assert(check_power_two(N));

    size_t size = N*sizeof(double);
    double *d_src,*d_dst;
    int n = 4;

    int threadsPerBlock = 512;
    int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CALL(cudaMalloc((void**)&d_src,size));
    CUDA_CALL(cudaMalloc((void**)&d_dst,size));

    CUDA_CALL(cudaMemcpy(d_src,t,size,cudaMemcpyHostToDevice));

    clock_t begin, end;

    begin = clock();
    while(n <= N)
    {
        gpu_idwt_pass<<<blocksPerGrid,threadsPerBlock>>>(d_src,d_dst,n);
        CUDA_CALL(cudaMemcpy(d_src,d_dst,n*sizeof(double),cudaMemcpyDeviceToDevice));
        n = n << 1;
    }
    cudaDeviceSynchronize();
    end = clock();
    CUDA_CALL(cudaMemcpy(t,d_src,size,cudaMemcpyDeviceToHost));
    
    printf("GPU Elapsed: %lfs \n", elapsed(begin,end));

    CUDA_CALL(cudaFree(d_src));
    CUDA_CALL(cudaFree(d_dst));
    return 0;
}

double cpu_dwt(double* t, int N)
{
    assert(check_power_two(N));
    int n=N;
    clock_t begin,end;
    double *tmp = (double*)malloc(N*sizeof(double));

    if(!tmp)
    {
        fprintf(stderr,"(host) cannot allocate memory for DWT\n");
        exit(EXIT_FAILURE);
    }

    begin = clock();
    while(n >= 4) 
    {
        cpu_d4_transform(t,tmp,n);
        memcpy(t,tmp,n*sizeof(double));

        n >>= 1;
    }

    end = clock();
    printf("CPU Elapsed: %lfs\n", elapsed(begin,end));
    free(tmp);
    return elapsed(begin,end);
}

double cpu_idwt(double *t, int N)
{
    assert(check_power_two(N));
    int n;
    clock_t begin, end; 

    double *tmp = (double*)malloc(N*sizeof(double));

    if(!tmp)
    {
        fprintf(stderr,"(host) cannot allocate memory for DWT\n");
        exit(EXIT_FAILURE);
    }

    begin = clock();
    for(n = 4; n <= N; n <<= 1)
    {
        cpu_d4_inv_transform(t,tmp,n);
        memcpy(t,tmp,n*sizeof(double));
    }
    end = clock();

    printf("CPU Elapsed: %lfs\n", elapsed(begin,end));
    free(tmp);

    return elapsed(begin,end);
}

int save_timing_forward()
{
    /*  benchmark CPU v. GPU on forward discrete wavelet transform 
        and save into csv file */

    FILE *save = fopen("results/timing.csv", "w+");

    if(!save) 
    {
        fprintf(stderr, "%s\n", "(host) unable to create timing file..");
        exit(EXIT_FAILURE);
    }

    fprintf(save,"%s,%s,%s\n", "N", "CPU_Time","GPU_Time");

    int n = 1<<10;
    double cpu_time,gpu_time;

    CUDA_CALL(cudaMemcpyToSymbol(g,_g,4*sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(h,_h,4*sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(ig,_ig,4*sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(ih,_ih,4*sizeof(double)));

    while(n <= (1<<24)) 
    {
        printf("n=%d\n",n);

        size_t size = n*sizeof(double);

        double * gpu_coef   = (double*)malloc(size);
        double * cpu_coef   = (double*)malloc(size);
        double * x0         = (double*)malloc(size);

        if(!gpu_coef || !cpu_coef || !x0) {
            fprintf(stderr, "%s\n", "could not allocate memory for signals!\n");
            exit(EXIT_FAILURE);
        }

        /* copy constants */

        fill_rand(x0, n);
        memcpy(gpu_coef,x0,size);
        memcpy(cpu_coef,x0,size); // save initial (random) array 

        gpu_time = gpu_dwt(gpu_coef,n);
        cpu_time = cpu_dwt(cpu_coef,n);

        if(!test_arrays_equal(gpu_coef,cpu_coef,n))
        {
            printf("Arrays not equal!\n");
            exit(EXIT_FAILURE);
        }

        fprintf(save, "%d,%lf,%lf\n", n,cpu_time,gpu_time);

        n <<= 1;

        free(cpu_coef);
        free(gpu_coef);
        free(x0);
    }

    fclose(save);
    return 0;
}

int test_dwt(const int N)
{
    double *signal = (double*)malloc(N*sizeof(double));

    fill_rand(signal, N);

    double * cpu_coef = (double*)malloc(N*sizeof(double));
    double * gpu_coef = (double*)malloc(N*sizeof(double));

    CUDA_CALL(cudaMemcpyToSymbol(g,_g,4*sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(h,_h,4*sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(ig,_ig,4*sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(ih,_ih,4*sizeof(double)));

    memcpy(cpu_coef,signal,N*sizeof(double));
    memcpy(gpu_coef,signal,N*sizeof(double));

    cpu_dwt(cpu_coef,N);
    gpu_dwt(gpu_coef,N);

    if(!test_arrays_equal(cpu_coef,gpu_coef,N))
    {
        printf("DWT not the same on CPU and GPU.\n");
        exit(EXIT_FAILURE);
    }
    printf("DWT test: pass.\n");
    return 0;
}

int test_idwt_cpu(const int N)
{
    double *signal = (double*)malloc(N*sizeof(double));

    fill_rand(signal, N);

    double * cpu_coef = (double*)malloc(N*sizeof(double));
    memcpy(cpu_coef,signal,N*sizeof(double));

    cpu_dwt(cpu_coef,N);
    cpu_idwt(cpu_coef,N);

    if(!test_arrays_equal(cpu_coef,signal,N))
    {
        printf("IDWT fail: signal not reconstructed on CPU.\n");
        exit(EXIT_FAILURE);
    }
    printf("IDWT CPU: pass.\n");
    return 0;
}

int test_idwt_gpu(const int N)
{
    double *signal = (double*)malloc(N*sizeof(double));

    fill_rand(signal, N);

    double * gpu_coef = (double*)malloc(N*sizeof(double));

    CUDA_CALL(cudaMemcpyToSymbol(g,_g,4*sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(h,_h,4*sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(ig,_ig,4*sizeof(double)));
    CUDA_CALL(cudaMemcpyToSymbol(ih,_ih,4*sizeof(double)));

    memcpy(gpu_coef,signal,N*sizeof(double));

    gpu_dwt(gpu_coef,N);
    gpu_idwt(gpu_coef,N);

    if(!test_arrays_equal(gpu_coef,signal,N))
    {
        printf("IDWT fail: signal not reconstructed on GPU.\n");
        exit(EXIT_FAILURE);
    }
    printf("IDWT GPU: pass.\n");
    return 0;
}

int main()
{
    int N = 1<<19;

    test_dwt(N);
    test_idwt_cpu(N);
    test_idwt_gpu(N);

    save_timing_forward();

    return 0;
}