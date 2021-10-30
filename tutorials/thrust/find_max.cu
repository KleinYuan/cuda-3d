#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
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
  float foo [N] = {0.1, 1.1, 2.1, 3.1, 5.1, -8.1};
  float *host_data = &foo[0];
  float *device_data;
  size_t size = N * sizeof(float);
  int max_idx = 0;

  cudaMallocManaged(&device_data, size);

  // Copy foo -> host_data to device_data
  cudaMemcpy(device_data, host_data, size, cudaMemcpyHostToDevice);
  // Find out the max index
  thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(device_data);
  thrust::device_vector<float>::iterator d_it = thrust::max_element(d_ptr, d_ptr + N);
  max_idx = d_it - (thrust::device_vector<float>::iterator)d_ptr;
  cudaDeviceSynchronize();
  // Sync
  checkCuda(cudaDeviceSynchronize());
  printf("Max Id is %d\n", max_idx);
  cudaFree(device_data);
}
