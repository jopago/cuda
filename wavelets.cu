#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

#define sqrt3 	1.73205080757
#define daub 	5.65685424949

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error %d at %s:%d\n",x, __FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

/* Device constants */

__constant__ const double h[4] = {
	(1 + sqrt3)/daub, (3 + sqrt3)/daub,
	(3 - sqrt3)/daub, (1 - sqrt3)/daub
};

__constant__ const double g[4] = {
	(1 - sqrt3)/daub, -(3 - sqrt3)/daub, (1 + sqrt3)/daub, -(3 + sqrt3)/daub
};

/* Host constants */

const double _h[4] = {
	(1 + sqrt3)/daub, (3 + sqrt3)/daub,
	(3 - sqrt3)/daub, (1 - sqrt3)/daub
};

const double _g[4] = {
	(1 - sqrt3)/daub, -(3 - sqrt3)/daub, (1 + sqrt3)/daub, -(3 + sqrt3)/daub
};

__global__ void gpu_dwt_pass(double *src, double *dest, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int half = n >> 1;

	if(2*i < (n-3)) {
		dest[i] 			= src[2*i]*h[0] + src[2*i+1]*h[1] + src[2*i+2]*h[2] + src[2*i+3]*h[3];
		dest[i+half] 		= src[2*i]*g[0] + src[2*i+1]*g[1] + src[2*i+2]*g[2] + src[2*i+3]*g[3];
	}
 	if(2*i == (n-2)) {
 		dest[i] 		= src[n-2]*h[0] + src[n-1]*h[1] + src[0]*h[2] + src[1]*h[3];
 		dest[i+half] 	= src[n-2]*g[0] + src[n-1]*g[1] + src[0]*g[2] + src[1]*g[3];
 	}

}

double elapsed(clock_t begin, clock_t end)
{
	return (double)(end - begin) / CLOCKS_PER_SEC;
}


int gpu_dwt(double *t, int n)
{
	size_t size = n*sizeof(double);
	double *d_src,*d_dst;

	int threadsPerBlock = 512;
    int blocksPerGrid =(n + threadsPerBlock - 1) / threadsPerBlock;

	CUDA_CALL(cudaMalloc((void**)&d_src,size));
	CUDA_CALL(cudaMalloc((void**)&d_dst,size));

	CUDA_CALL(cudaMemcpy(d_src,t,size,cudaMemcpyHostToDevice));
	printf("%d\n",size);

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

void cpu_d4transform(double *t, const int n)
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

		for (i = 0; i < half; i++) {
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

void cpu_dwt(double* t, int n)
{
	while(n >= 4) {
		cpu_d4transform(t,n);
		n >>= 1;
	}
}

void disp(double *t, int n)
{
	int i;
	for(i=0;i<n;i++)
	{
		printf("%lf ", t[i]);
	}
	printf("\n");
}



void fill_rand(double *t, int n)
{
	int i;
	for(i=0;i<n;i++)
	{
		t[i] = (rand() % INT_MAX)*1e-5;
	}
}

int main()
{
	const int N = (1<<20);
	size_t size = N*sizeof(double);
	clock_t begin, end; 
	double * t 	= (double*)malloc(size);
	double * t2 = (double*)malloc(size);

	if(!t || !t2) {
		fprintf(stderr, "%s\n", "could not allocate memory for signals!\n");
		exit(EXIT_FAILURE);
	}

	fill_rand(t, N);
	memcpy(t2,t,size);

	gpu_dwt(t,N);


	begin = clock();
	cpu_dwt(t2,N);
	end = clock();

	printf("CPU elapsed: %lfs\n", elapsed(begin,end));

	int i;
	for(i=0;i<N;i++) 
	{
		if(fabs(t[i] - t2[i]) > 1e-4) 
		{
			printf("results do not match!\n");
			exit(EXIT_FAILURE);
		}
	}


	printf("results are the same!\n"); 
	exit(EXIT_SUCCESS);
}