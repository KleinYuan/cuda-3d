#include <cublas_v2.h>
#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess){
      fprintf(stderr, "CUDA Runtime Error: %s \n", cudaGetErrorString(result));
  }
  return result;
}

// https://docs.nvidia.com/cuda/cublas/index.html
int main()
{
  // Init
  const int N = 6;
  float foo [N] = {0.1, 1.1, 2.1, 3.1, 5.1, 4.1};
  float *host_data = &foo[0];
  float *device_data;
  size_t size = N * sizeof(float);
  int max_idx = 0;
  cublasHandle_t cublas_handle_;
  cublasCreate(&cublas_handle_);

  cudaMallocManaged(&device_data, size);

  // Copy foo -> host_data to device_data
  cudaMemcpy(device_data, host_data, size, cudaMemcpyHostToDevice);
  // Find out the max index
  cublasIsamax(cublas_handle_, N, device_data, 1, &max_idx);
  // Sync
  checkCuda(cudaDeviceSynchronize());
  printf("Max Id is %d\n", max_idx);
  cudaFree(device_data);
  cublasDestroy(cublas_handle_);
}
