#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess){
      fprintf(stderr, "CUDA Runtime Error: %s \n", cudaGetErrorString(result));
  }
  return result;
}

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void findMaxKernel(float *data, int N, float *max_val){
  int idxWithinTheGrid = threadIdx.x + blockDim.x * blockIdx.x;
  int gridStride = gridDim.x * blockDim.x;

  for(int i = idxWithinTheGrid; i < N; i += gridStride)
  {
    float old = atomicMax(max_val, data[i]);
    printf("[findMaxKernel] Perform GPU code, comparing %f (old)  and %f (new) at %p \n", old, data[i], &data[0] + sizeof(float) * i);
  }
}

__global__ void findMaxAndOffsetKernel(float *data, int N, float *max_val, float *device_offset){
  int idxWithinTheGrid = threadIdx.x + blockDim.x * blockIdx.x;
  int gridStride = gridDim.x * blockDim.x;

  for(int i = idxWithinTheGrid; i < N; i += gridStride)
  {
    float old = atomicMax(max_val, data[i]);
    float old_offset = old + *device_offset;
    printf("[findMaxKernel] Perform GPU code, comparing %f (old)  and %f (new) at %p \n", old, data[i], &data[0] + sizeof(float) * i);
    printf("[findMaxKernel] After offset %f (old)  becomes %f (old_offset)  \n", old, old_offset);

  }
}

int main()
{
  // Init
  const int N = 4;
  float foo [N] = {0.1, 1.1, 2.1, 3.1};
  float *host_data = &foo[0];
  float *device_data;
  size_t size = N * sizeof(float);
  float max_val = 0.f;
  float *device_max;

  // The following offset is to show how to de-reference device pointer in a kernel
  // uncomment findMaxAndOffsetKernel to see the effect
  float offset_val = 2.f;
  float *device_offset;
  cudaMalloc(&device_offset, sizeof(float));
  cudaMemcpy(device_offset, &offset_val,  sizeof(float), cudaMemcpyHostToDevice);


  cudaMallocManaged(&device_data, size);
  cudaMallocManaged(&device_max, sizeof(float));
  // Copy foo -> host_data to device_data
  cudaMemcpy(device_data, host_data, size, cudaMemcpyHostToDevice);
  cudaMemcpy(device_max, &max_val, sizeof(float), cudaMemcpyHostToDevice);
  // Run a custom kernel to find the max val in foo, adding the index to the element
  findMaxKernel<<<4, 1>>>(device_data, N, device_max);
  // findMaxAndOffsetKernel<<<4, 1>>>(device_data, N, device_max, device_offset);
  cudaMemcpy(&max_val, device_max, sizeof(float), cudaMemcpyDeviceToHost);
  // Sync
  checkCuda(cudaDeviceSynchronize());
  printf("max_val is %lf\n", max_val);
  cudaFree(device_data);
  cudaFree(device_max);
  cudaFree(device_offset);
}
