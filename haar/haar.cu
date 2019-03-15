#include <cuda_runtime.h>
#include "../common/cuda_ptr.h"
#include "../common/utils.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <iostream>

// 1.0/ sqrt(2)
#define haar 0.707106781f

/*  Haar wavelets forward horizontal and vertical passes 
    To get the full decomposition we apply one after the other
    log_2(N) times and its done */

template<typename T>
__global__ void gpu_haar_horizontal(T* in, const int n, T* out, const int N)
{
	auto i = blockIdx.x * blockDim.x + threadIdx.x;
	auto j = blockIdx.y * blockDim.y + threadIdx.y;

	if( i < n && j < n/2)
	{
		auto idx_in 	= i*N + 2*j;    // (i,2*j)
		auto idx_out 	= j + i*N;      // (i,j)

		out[idx_out] 		= haar*(in[idx_in] + in[idx_in+1]);
        // out(i,2*j + n/2)
		out[idx_out + n/2] 	= haar*(in[idx_in] - in[idx_in+1]);
	}
}

template<typename T>
__global__ void gpu_haar_vertical(T* in, const int n, T* out, const int N)
{
	auto i = blockIdx.x * blockDim.x + threadIdx.x;
	auto j = blockIdx.y * blockDim.y + threadIdx.y;

	if( i < n/2 && j < n)
	{
		auto in_idx_1 	= 2*i*N + j;
		auto in_idx_2 	= (2*i+1)*N + j;
		auto out_idx 	= j + i*N;

        out[out_idx]            = haar*(in[in_idx_1] + in[in_idx_2]);
        // out(i+n/2,j)
        out[out_idx + N*n/2]    = haar*(in[in_idx_1] - in[in_idx_2]);
	}
}

void mat_to_double(cv::Mat in, double* out)
{
    for(auto i = 0; i < in.rows; i++)
    {
        for(auto j = 0; j < in.rows; j++)
        {
            out[i*in.rows + j] = in.at<unsigned char>(j,i);
        }
    }
}

void double_to_mat(double *in, cv::Mat* out)
{
    for(auto i = 0; i < out->rows; i++)
    {
        for(auto j = 0; j < out->rows; j++)
        {
            out->at<unsigned char>(j,i) = 10*fabs(in[i*out->rows + j]);
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
    const int size = N*N*sizeof(double);
    cv::Mat frame;
    cv::Mat channels[3]; 
    cv::Size frame_size(N,N);
    double *frame_buf   = new double[N*N];
    double *wav_buf     = new double[N*N];

    cv::namedWindow("window", 1);

    cuda_ptr<double> gpu_frame(frame_buf,size);
    cuda_ptr<double> gpu_wav(frame_buf,size);

    int blockWidth = 8;
    dim3 dimBlock(blockWidth,blockWidth);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y); // 64 threads (pixels) per block
    int screenshot = 0;

    for(;;)
    {
        cap >> frame;

        // cv::cvtColor(frame,frame,CV_RGB2GRAY);

        cv::resize(frame,frame,frame_size); 
        cv::split(frame, channels);

        for(auto i = 0; i < 3; i++)
        {
            frame = channels[i]; 

            mat_to_double(frame, frame_buf);
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

            double_to_mat(wav_buf, &channels[i]); 
        }

        cv::merge(channels, 3, frame);
        cv::imshow("window",frame);

        if(!screenshot)
        {
            screenshot = 1;
            cv::imwrite("img/decomposition_frame.png", frame);
        }

        if(cv::waitKey(30) >= 0) break;
    }

	return 0;
}