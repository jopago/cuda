#ifndef _UTILS_H
#define _UTILS_H

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error %s (%d) at %s:%d\n", cudaGetErrorString(x),x, __FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

double elapsed(clock_t begin, clock_t end)
{
    return (double)(end - begin) / CLOCKS_PER_SEC;
}

int check_power_two(const int n)
{
    return !(n & (n-1));
}

void disp(double *t, const int n)
{
    int i;
    for(i=0;i<n;i++)
    {
        printf("%lf ", t[i]);
    }
    printf("\n");
}

void fill_rand(double *t, const int n)
{
    int i=0;
    for(;i<n;i++)
    {
        t[i] = ((double)rand())/INT_MAX;
    }
}

void fill_ones(double *t, const int n)
{
    int i=0;
    for(;i<n;i++)
    {
        t[i] = 1.;
    }
}

void fill_rand_2d(double *t, int n)
{
    int i;

    for(i=0;i<n*n;i++)
    {
        t[i] = (double)rand() / INT_MAX;
    }
}

void disp_2d(double *t, int n)
{
    int i,j;

    for(i=0;i<n;i++)
    {
        for(j=0;j<n;j++)
        {
            printf("%lf ", t[i*n+j]);
        }
        printf("\n");
    }
    
}

int test_arrays_equal(double *t1, double *t2, const int n,
    const double tol = 1e-6)
{
    int i=0;
    for(;i<n;i++) 
    {
        if(fabs(t1[i] - t2[i]) > tol) 
        {
            // printf("Arrays not equal!\n");
            return 0;
        }
    }

    // printf("Arrays are equal!\n");
    return 1;
}

#endif