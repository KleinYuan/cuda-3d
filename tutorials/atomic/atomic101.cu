#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess){
      fprintf(stderr, "CUDA Runtime Error: %s \n", cudaGetErrorString(result));
  }
  return result;
}

__global__ void atomicAddIdx(float *data, int N, int start_idx, int stop_idx){
  int idxWithinTheGrid = threadIdx.x + blockDim.x * blockIdx.x;
  int gridStride = gridDim.x * blockDim.x;

  for(int i = idxWithinTheGrid; i < N; i += gridStride)
  {
    if (i < start_idx or i >= stop_idx){
        continue;
    }
    // atomicAdd accepts the address of each element
    // This is wrong: atomicAdd(&data[0] + sizeof(float) * i, i); since atomicAdd
    // takes in a pointer instead of the actual address
    atomicAdd(data + i, i);
    printf("[atomicAddIdx] Perform GPU code %f + %d...at address: %p \n", data[i], i*i, &data[0] + sizeof(float) * i);
  }
}

__global__ void atomicAddIdxSquared(float *data, int N, int start_idx, int stop_idx){
  int idxWithinTheGrid = threadIdx.x + blockDim.x * blockIdx.x;
  int gridStride = gridDim.x * blockDim.x;

  for(int i = idxWithinTheGrid; i < N; i += gridStride)
  {
    if (i < start_idx or i >= stop_idx){
        continue;
    }
    // atomicAdd accepts the address of each element
    atomicAdd(data + i, i*i);
    printf("[atomicAddIdxSquared] Perform GPU code %f + %d...at address: %p \n", data[i], i*i, &data[0] + sizeof(float) * i);
  }
}


int main()
{
  // Init
  const int N = 4;
  float foo [N] = {0.1, 1.1, 2.1, 3.1};
  float bar [N] = {0., 0., 0., 0.};
  float *host_data = &foo[0];
  float *device_data;
  float *new_host_data = &bar[0];
  size_t size = N * sizeof(float);

  cudaMallocManaged(&device_data, size);

  // Copy foo -> host_data to device_data
  cudaMemcpy(device_data, host_data, size, cudaMemcpyHostToDevice);
  // Perform atomicAddIdx, adding the index to the element
  atomicAddIdx<<<4, 1>>>(device_data, N, 0, 2);
  // atomicAddIdx<<<4, 1>>>(device_data, N, 0, 4); // replace above with this to see the behavior of concurrent update
  // Perform atomicAddIdxSquared, adding the index * index to the element
  atomicAddIdxSquared<<<4, 1>>>(device_data, N, 2, 4);
  // atomicAddIdxSquared<<<4, 1>>>(device_data, N, 0, 4); // replace above with this to see the behavior of concurrent update
  // Sync
  checkCuda(cudaDeviceSynchronize());
  // Copy back to host to print out
  cudaMemcpy(new_host_data, device_data, size, cudaMemcpyDeviceToHost);
  for (int i=0; i < N; i++) {
    printf("Updated number on %d is %lf\n", i, new_host_data[i]);
  }
  cudaFree(device_data);
}
