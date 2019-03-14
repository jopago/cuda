#ifndef _CUDA_PTR
#define _CUDA_PTR

// host wrapping of CUDA pointer allocation 

#define CUDA_CALL_VOID(x) do { if((x) != cudaSuccess) { \
    printf("Error %s (%d) at %s:%d\n", cudaGetErrorString(x),x, __FILE__,__LINE__); \
    exit(EXIT_FAILURE);}} while(0)

template<typename T>
class cuda_ptr
{
public:
	cuda_ptr(const T* host_ptr, const int size) 
	{
		CUDA_CALL_VOID(cudaMalloc(&ptr,size));
		CUDA_CALL_VOID(cudaMemcpy(ptr,host_ptr,size,cudaMemcpyHostToDevice));
	}

	cuda_ptr(const int size)
	{
		CUDA_CALL_VOID(cudaMalloc(&ptr,size));
	}

	~cuda_ptr() 
	{
		CUDA_CALL_VOID(cudaFree(ptr));
	}

	void to_host(T* host_ptr, const int size)
	{
		CUDA_CALL_VOID(cudaMemcpy(host_ptr,ptr,size,cudaMemcpyDeviceToHost));
	}

	T* devptr()
	{
		return ptr;
	}

	T operator[](int i)
	{
		return ptr[i];
	}
private:
	T* ptr; // pointer to device memory 
};

#endif