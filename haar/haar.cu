#include <cuda_runtime.h>
#include "../common/cuda_ptr.h"
#include "../common/utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

// 1.0/ sqrt(2)
#define haar 0.5f

/*  Haar wavelets forward horizontal and vertical passes 
    To get the full decomposition we apply one after the other
    log_2(N) times and its done */

template<typename T>
__global__ void gpu_haar_horizontal(T* in, const int n, T* out, const int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if( i < n && j < n/2)
	{
		int idx_in 	= i*N + 2*j;    // (i,2*j)
		int idx_out 	= j + i*N;      // (i,j)

		out[idx_out] 		= haar*(in[idx_in] + in[idx_in+1]);
        // out(i,2*j + n/2)
		out[idx_out + n/2] 	= haar*(in[idx_in] - in[idx_in+1]);
	}
}

template<typename T>
__global__ void gpu_haar_vertical(T* in, const int n, T* out, const int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if( i < n/2 && j < n)
	{
		int in_idx_1 	= 2*i*N + j;
		int in_idx_2 	= (2*i+1)*N + j;
		int out_idx 	= j + i*N;

        out[out_idx]            = haar*(in[in_idx_1] + in[in_idx_2]);
        // out(i+n/2,j)
        out[out_idx + N*n/2]    = haar*(in[in_idx_1] - in[in_idx_2]);
	}
}

template<typename T>
__global__ void gpu_inverse_haar_vertical(T* in, const int h, const int w, T* out, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < h && j < w)
    {
        int out_idx_1 = 2*i*N + j;
        int out_idx_2 = (2*i+1)*N + j;
        int in_idx_1 = i*N + j;
        int in_idx_2 = (i+h)*N + j;

        out[out_idx_1] = (in[in_idx_1] + in[in_idx_2]);
        out[out_idx_2] = (in[in_idx_1] - in[in_idx_2]);
    }
}

template<typename T>
__global__ void gpu_inverse_haar_horizontal(T* in, const int h, const int w, T* out, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < h && j < w)
    {
        int out_idx_1 = i*N + 2*j;
        int out_idx_2 = i*N + 2*j+1;
        int in_idx_1 = i*N + j;
        int in_idx_2 = i*N + j + w;

        out[out_idx_1] = (in[in_idx_1] + in[in_idx_2]);
        out[out_idx_2] = (in[in_idx_1] - in[in_idx_2]);
    }
}

template<typename T>
__global__ void gpu_low_pass(T* x, const int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < n && j < n)
    {
        if(fabs(x[i*n+j]) < 1.5f)
        {
            x[i*n+j] = 0.0f;
        }
    }
}


void mat_to_float(cv::Mat in, float* out)
{
    for(int i = 0; i < in.rows; i++)
    {
        for(int j = 0; j < in.rows; j++)
        {
            out[i*in.rows + j] = in.at<float>(j,i);
        }
    }
}

void float_to_mat(float *in, cv::Mat* out)
{
    for(int i = 0; i < out->rows; i++)
    {
        for(int j = 0; j < out->rows; j++)
        {
            out->at<float>(j,i) = 0.1*fabs(in[i*out->rows + j]);
        }
    }
}

void float_to_mat_2(float *in, cv::Mat* out)
{
    for(int i = 0; i < out->rows; i++)
    {
        for(int j = 0; j < out->rows; j++)
        {
            out->at<float>(j,i) = 0.01*in[i*out->rows + j];
        }
    }
}

int main()
{
    cv::VideoCapture cap(0);

    if(!cap.isOpened()) 
    {
        std::cout << "could not open camera" << std::endl;
        return -1;
    }

    const int N = 512;
    const int size = N*N*sizeof(float);
    
    cv::Size frame_size(N,N);
    cv::Mat frame(frame_size,CV_32FC1);
    cv::Mat inverse(frame_size,CV_32FC1);

    float *frame_buf   = new float[N*N];
    float *wav_buf     = new float[N*N];

    cv::namedWindow("Wavelets", 1);
    cv::namedWindow("Inverse",1);

    cuda_ptr<float> gpu_frame(frame_buf,size);
    cuda_ptr<float> gpu_wav(frame_buf,size);

    int blockWidth = 8;
    dim3 dimBlock(blockWidth,blockWidth);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y); // 64 threads (pixels) per block
    int screenshot = 0;

   
    for(;;)
    {
        cap >> frame;

        frame.convertTo(frame,CV_32FC1);
        cv::cvtColor(frame,frame,CV_RGB2GRAY);

        cv::resize(frame,frame,frame_size); 
        mat_to_float(frame, frame_buf);
        gpu_frame.copy(frame_buf, size);
        // compute wavelet transform of current frame 
        
        for(auto n = N; n > 1; n /= 2)
        {
            // Perform horizontal then vertical forward pass 
            gpu_haar_horizontal<<<dimGrid,dimBlock>>>(gpu_frame.devptr(),n,
                gpu_wav.devptr(), N);
            gpu_haar_vertical<<<dimGrid,dimBlock>>>(gpu_wav.devptr(),n,
                gpu_frame.devptr(), N);
            CUDA_CALL(cudaDeviceSynchronize());
        }

        // write back to CPU
        gpu_frame.to_host(wav_buf,size);
        float_to_mat(wav_buf, &frame); 
       
        cv::imshow("Wavelets",frame);

        /* Filtering */

        // gpu_low_pass<<<dimGrid,dimBlock>>>(gpu_frame.devptr(), N);
        // CUDA_CALL(cudaDeviceSynchronize());

        /* Inverse wavelet transform */

        auto k = 1;
        auto height = 1;
        auto width = 1;

        while(k < N)
        {
            gpu_inverse_haar_vertical<<<dimGrid,dimBlock>>>(gpu_frame.devptr(),height,
                width,
                gpu_wav.devptr(), N);
            height *= 2;

            gpu_inverse_haar_horizontal<<<dimGrid,dimBlock>>>(gpu_wav.devptr(),height,
                width,
                gpu_frame.devptr(), N);

            width *= 2;
            CUDA_CALL(cudaDeviceSynchronize());

            k *= 2;
        }

        gpu_frame.to_host(wav_buf,size); 
        float_to_mat_2(wav_buf,&frame);
        
        cv::imshow("Inverse",frame); 
        
        if(!screenshot)
        {
            screenshot = 1;
            cv::imwrite("img/decomposition_frame.png", frame);
        } 

        if(cv::waitKey(30) >= 0) break; 
    }

	return 0;
}
